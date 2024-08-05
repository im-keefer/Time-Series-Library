import datetime
import pandas as pd
import pytz
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, loss_visual
from utils.losses import vrmse_loss, vmse_loss
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import csv
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'VRMSE':
            return vrmse_loss()
        elif loss_name == 'VMSE':
            return vmse_loss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_stamp, batch_y_stamp) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if (self.args.features == 'MS' or self.args.features == 'MT') else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        train_loss_pts = []
        val_loss_pts = []

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_stamp, batch_y_stamp) in enumerate(train_loader):
                print('a', batch_x.shape)
                print('b', batch_x_mark.shape)
                print('c', batch_y.shape)
                print('c', batch_y_mark.shape)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if (self.args.features == 'MS' or self.args.features == 'MT') else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if (self.args.features == 'MS' or self.args.features == 'MT') else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            train_loss_pts.append(train_loss)
            val_loss_pts.append(vali_loss)
            early_stopping(vali_loss, self.model, model_optim, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Plot training and validation loss
        self.folder_path = './test_results/' + setting + '/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        else: # Force output into new folder
            i = 1
            self.folder_path = './test_results/' + setting + str(i) + '/'
            while os.path.exists(self.folder_path):
                i = i + 1
                self.folder_path = './test_results/' + setting + str(i) + '/'
            os.makedirs(self.folder_path)

        loss_visual(train_loss_pts, val_loss_pts, os.path.join(self.folder_path, 'loss_plot.pdf'))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        #folder_path = './test_results/' + setting + '/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.model.eval()
        batch_mse = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_stamp, batch_y_stamp) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if (self.args.features == 'MS' or self.args.features == 'MT') else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    if shape[0] == 1:
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                    else: # We have multiple batches, inverse them all why not
                        for j in range(shape[0]):
                            outputs[j] = test_data.inverse_transform(outputs[j]).reshape(outputs[j].shape)
                            batch_y[j] = test_data.inverse_transform(batch_y[j]).reshape(outputs[j].shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                mse_for_batch = metric(pred[0, :, -1], true[0, :, -1])[1]
                for j in range(len(batch_x_mark[0, :, -1])):
                    batch_mse.append([batch_x_mark[0, j, 0].item(), mse_for_batch])

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        if shape[0] == 1:
                            input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                        else:
                            for j in range(shape[0]):
                                input[j] = test_data.inverse_transform(input[j]).reshape(input[j].shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(self.folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        f = open(self.folder_path + "loss_into_day.csv", 'w', newline='')
        writer = csv.writer(f)
        writer.writerows(batch_mse)
        f.close()

        np.save(self.folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(self.folder_path + 'pred.npy', preds)
        np.save(self.folder_path + 'true.npy', trues)

        return

    def inference(self, setting, inference_path, data_path):
        test_data, test_loader = self._get_data(flag='inference')
        print('device: ', self.device)
        print('loading model', inference_path)
        self.model.load_state_dict(torch.load(inference_path, map_location=torch.device(self.device)))
        self.model.to(self.device)

        preds = []
        trues = []
        stamps = []

        self.folder_path = './inference_results/' + setting + '/'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        else: # Force output into new folder
            i = 1
            self.folder_path = './inference_results/' + setting + str(i) + '/'
            while os.path.exists(self.folder_path):
                i = i + 1
                self.folder_path = './inference_results/' + setting + str(i) + '/'
            os.makedirs(self.folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_stamp, batch_y_stamp) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if (self.args.features == 'MS' or self.args.features == 'MT') else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    if shape[0] == 1:
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                    else: # We have multiple batches, inverse them all why not
                        for j in range(shape[0]):
                            outputs[j] = test_data.inverse_transform(outputs[j]).reshape(outputs[j].shape)
                            batch_y[j] = test_data.inverse_transform(batch_y[j]).reshape(outputs[j].shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y
                stamp = batch_y_stamp[:, -self.args.pred_len:]
                stamp2 = batch_y_stamp
                #print('stamp2: ', stamp2.shape)
                if i == 0:
                    torch.set_printoptions(precision=9, sci_mode=False)
                    print('stamp2: ', stamp2)
                    print('a: ', stamp2[0][0], stamp2[0][-50:])

                preds.append(pred)
                trues.append(true)
                stamps.append(stamp)
                if False and (i == 0 or (i > 2329 and i < 2335)):
                    print('[0]: ', i+2, '|   ', stamp[0][0].item(), true[0][0][0], pred[0][0][0])
                    print('[49]: ', i+2, '|   ', stamp[0][49].item(), true[0][49][0], pred[0][49][0])
                    print('Finished inferencing batch ', i+1)

        preds = np.array(preds)
        trues = np.array(trues)
        stamps = np.array(stamps, dtype=int)
        print('test shape:', preds.shape, trues.shape, stamps.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        stamps = stamps.reshape(-1, stamps.shape[-1])

        preds = np.squeeze(preds, axis=-1)
        trues = np.squeeze(trues, axis=-1)
        print('test shape:', preds.shape, trues.shape, stamps.shape)
        
        flat_preds = preds.flatten()
        flat_trues = trues.flatten()
        flat_stamps = stamps.flatten().astype(int)
        np.set_printoptions(precision=20)
        print('pred, trues, stamps len: ', len(flat_preds), len(flat_trues), len(flat_stamps))

        num_sequences, sequence_length = stamps.shape
        distances = np.tile(np.arange(sequence_length, dtype=int), num_sequences).astype(int)

        def unix_to_string(unix_timestamp):
            # Define the timezone that handles daylight saving time (Eastern Time)
            timezone = pytz.timezone('America/New_York')
            
            # Convert the Unix timestamp to a datetime object in UTC
            dt_utc = datetime.datetime.fromtimestamp(unix_timestamp, tz=datetime.timezone.utc)
            
            # Convert the datetime object to the specified timezone, which handles DST
            dt_local = dt_utc.astimezone(timezone)
            
            # Format the datetime object to the desired string format
            dt_str = dt_local.strftime("%d.%m.%Y %H:%M:%S.%f")[:-3]  # Trim to milliseconds
            timezone_str = dt_local.strftime(" GMT%z")
            
            # Create the final formatted string
            formatted_string = f"{dt_str}{timezone_str}"
            
            return formatted_string
        
        vectorized_unix_to_string = np.vectorize(unix_to_string)
        formatted_stamps = vectorized_unix_to_string(flat_stamps)

        data_csv_df = pd.read_csv(data_path)

        open_values = data_csv_df[data_csv_df['Local time'].isin(formatted_stamps)]['Open'].values
        not_in_values = data_csv_df[~data_csv_df['Local time'].isin(formatted_stamps)]['Open'].values

        print('open', len(open_values), len(not_in_values), len(data_csv_df))
        #print('a', )
        #print('b', data_csv_df['Local time'].iloc[0], 
        #    data_csv_df[data_csv_df['Local time'].isin(formatted_stamps)]['Open'].iloc[0], 
        #    data_csv_df[~data_csv_df['Local time'].isin(formatted_stamps)]['Open'].iloc[0])
        print('flat_stamps', len(formatted_stamps), formatted_stamps[0])

        df = pd.DataFrame({
            'Timestamp': flat_stamps,
            'True': open_values,
            'Pred': flat_preds,
            'Distance': distances
        })


        df.to_csv(self.folder_path + 'results.csv', index=False)

        print(f"CSV written to {self.folder_path + 'results.csv'}")

        return