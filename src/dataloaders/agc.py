import pyarrow as pa
import pyarrow.compute as pc
from torch.utils.data import Dataset, DataLoader
import torch
import os
import warnings
from torch.nn import Conv2d
import numpy as np
warnings.filterwarnings("ignore")

from src.dataloaders.base import SequenceDataset, default_data_path

class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data * std) + mean


class SignalNormScaler:
    def __init__(self):
        pass

    def scale_mz(self, raw_data_full, centers_full):
        batch_size = raw_data_full.shape[0]  # batch, timestep, 2, maxlen
        time_step = raw_data_full.shape[1]
        centers = centers_full.reshape((batch_size, 1, 1))
        raw_data_full[:, :, 0] = raw_data_full[:, :, 0] - centers

    def scale_intens(self, raw_data_full, factors_full):
        batch_size = raw_data_full.shape[0]  # batch, timestep, 2, maxlen
        time_step = raw_data_full.shape[1]
        factors = factors_full.reshape((batch_size, time_step, 1))
        raw_data_full[:, :, 0] = raw_data_full[:, :, 0] / factors


def _split_into_subsequences(sequence, subsequence_length):
    return [sequence[i:i+subsequence_length] for i in range(0, len(sequence), subsequence_length)]

def _pad_sequences(sequences, max_length, padding_value=0):
    return [seq + [padding_value]*(max_length - len(seq)) for seq in sequences]

def _getWithinWindows(mzs, intens, startMz, endMz, offset = 4):
    mzs_intens = list(zip(mzs, intens))
    mzs_intens.sort(key=lambda x: x[0])
    j_start = max(startMz - offset, min(mzs))
    j_end = min(endMz + offset, max(mzs))
    array = list(filter(lambda x: x[0] >= j_start and x[0] <= j_end, mzs_intens))
    if(len(array) == 0):
        return [0],[0]
    mz, inten = list(zip(*array))
    return mz, inten

class MassSpecSequenceDataset(SequenceDataset):

    @property
    def n_tokens_time(self):
        # Shape of the dates: depends on `timeenc` and `freq`
        return self.dataset_train.n_tokens_time  # 

    @property
    def d_input(self):
        return self.dataset_train.d_input



    @property
    def l_output(self):
        return self.dataset_train.pred_len

    def _get_data_filename(self, variant):
        return self.variants[variant]

    _collate_arg_names = ["mark", "mask"] # Names of the two extra tensors that the InformerDataset returns

    def setup(self):
        self.data_dir = self.data_dir or default_data_path

        self.dataset_train = self._dataset_cls(
        root_path=self.data_dir,
        flag="train",
        size=self.size,
        offset=self.offset,
        data_path=self._get_data_filename(self.variant),
        target=self.target,
        scale_raw = self.scale_raw,
        scale_meta = self.scale_meta,
        maxIT = self.maxIT,
        meta_features = self.meta_features,
        timestep=self.timestep
        )

        self.dataset_val = self._dataset_cls(
            root_path=self.data_dir,
            flag="val",
            size=self.size,
            offset=self.offset,
            data_path=self._get_data_filename(self.variant),
            target=self.target,
            scale_raw = self.scale_raw,
            scale_meta = self.scale_meta,
            maxIT = self.maxIT,
            meta_features = self.meta_features,
            timestep=self.timestep
        )

        self.dataset_test = self._dataset_cls(
            root_path=self.data_dir,
            flag="test",
            size=self.size,
            offset=self.offset,
            data_path=self._get_data_filename(self.variant),
            target=self.target,
            scale_raw = self.scale_raw,
            scale_meta = self.scale_meta,
            maxIT = self.maxIT,
            meta_features = self.meta_features,
            timestep=self.timestep
        )





class MassSpecDataset(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        offset=4,
        data_path="BSA_pAGC_training.arrow",
        target="filltime",
        scale_raw=True,
        scale_meta=True,
        maxIT=500,
        meta_features=True,
        timestep=5
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 32 * 4 * 4
            self.label_len = 32 * 1
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.offset = offset
        self.target = target
        self.scale_raw = scale_raw
        self.scale_meta = scale_meta
        self.maxIT = maxIT
        self.meta_features = meta_features
        self.timestep = timestep
        #self.freq = freq
        #self.eval_stamp = eval_stamp
        #self.eval_mask = eval_mask
        #self.forecast_horizon = self.pred_len

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def _borders(self, df_raw):
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        return border1s, border2s


    def __read_data__(self):
        self.raw_scaler = SignalNormScaler()
        self.meta_scaler = StandardScaler()

        with pa.memory_map(os.path.join(self.root_path, self.data_path), 'r') as source:
            loaded_arrays = pa.ipc.open_file(source).read_all()

        only_pAGC = loaded_arrays.filter(pc.match_substring_regex(pc.field("scanType"), "ITMS"))
        only_SIM = loaded_arrays.filter(pc.match_substring_regex(pc.field("scanType"), "SIM"))

        _pagc_groupIndex_itms = []
        for i in range(only_pAGC.num_rows):
            _pagc_groupIndex_itms.append(dict(only_pAGC['scanHeader'][i])['API PAGC Scan Group Index'])
        only_pAGC = only_pAGC.append_column("pagcScanIndex", [_pagc_groupIndex_itms])

        raw_predictors = []
        raw_pagc_filltime = []
        raw_pagc_tics = []
        meta_predictors = []

        predictions = []
        for row in range(1000): #range(only_SIM.num_rows):
            meta = []
            curRow = only_SIM.take([row])
            iit = curRow['iit'][0].as_py()
            if(iit >= self.maxIT):
                continue
            pagcEnd = curRow['scanBasedOn'][0]
            pagcScanIndex = dict(only_SIM.take([0])['scanHeader'][0])['API PAGC Scan Group Index']
            only_SIM_with_scanIndex = only_pAGC.filter(pc.match_substring_regex(pc.field("pagcScanIndex"), pagcScanIndex))
            pagcEnd = int(pc.index(only_SIM_with_scanIndex['scanNumber'], pagcEnd).as_py())
            startMz = curRow['segmented'][0][0][0].as_py()
            endMz = curRow['segmented'][0][0][1].as_py()
            meta = [(endMz+startMz)/2, (endMz-startMz), 
                                    only_SIM_with_scanIndex['retentionTime'][pagcEnd].as_py()
            ]
                                    # curRow['agcs'][0].as_py(), 
                                    #float(dict(curRow['scanHeader'][0])["RawOvFtT"]) ] ) #center, width, agc, rawovftt GLOBAL

            if (pagcEnd < self.timestep-1 ):
                startIndex = 0
                endIndex = pagcEnd+1
            else:
                startIndex = pagcEnd-self.timestep+1
                endIndex = pagcEnd+1

            mzsRaw = only_SIM_with_scanIndex['masses'][startIndex:endIndex]
            intensRaw = only_SIM_with_scanIndex['intensities'][startIndex:endIndex]
            mzs = []
            intens = []
            pagc_filltimes = []
            pagc_tics = []
            for spec in range(len(mzsRaw)):
                res = _getWithinWindows(list(mzsRaw[spec].as_py()), list(intensRaw[spec].as_py()), 
                                       float(startMz), float(endMz), offset = self.offset)
                mzs.append(res[0])
                intens.append(res[1])
                pagc_filltimes = [float(x.as_py()[3][1])
                                    for x in only_SIM_with_scanIndex["scanHeader"][startIndex:endIndex]]
                pagc_tics = [x.as_py() for x in only_SIM_with_scanIndex["TIC"][startIndex:endIndex]]
            if(len(intens) == 0):
                continue
            max_len = 128
            
            _mzsInputs=[]
            _intensInputs = []
            _pagc_fts = []
            _pagc_tics = []
            for ind, arr in enumerate(intens):
 
                mzsInput = _split_into_subsequences(list(mzs[ind]),max_len)

                mzsInput = _pad_sequences(mzsInput, max_len, padding_value = 0)
                intensInput = _split_into_subsequences(list(intens[ind]), max_len)
                intensInput = _pad_sequences(intensInput, max_len, padding_value = 0)
                pagc_filltimesInput = [pagc_filltimes[ind] for _ in range(len(intensInput))]
                pagc_ticsInput = [pagc_tics[ind] for _ in range(len(intensInput))]

                _mzsInputs.extend(mzsInput)
                _intensInputs.extend(intensInput)
                _pagc_fts.extend(pagc_filltimesInput)
                _pagc_tics.extend(pagc_ticsInput)
            meta_predictors.append(meta)
        
            raw_pagc_filltime.append(_pagc_fts[-self.timestep:])
            raw_pagc_tics.append(_pagc_tics[-self.timestep:])
            predictions.append(iit)

            xes = np.array(list(zip(_mzsInputs[-self.timestep:],_intensInputs[-self.timestep:]))) # mz, intens
            xes = xes.reshape((self.timestep, 2, max_len))
            raw_predictors.append(xes)

        border1s, border2s = self._borders(raw_predictors)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        centers = np.array(meta_predictors)[:, 0]

        raw_predictors = np.array(raw_predictors)

        raw_pagc_filltime = np.array(raw_pagc_filltime)
        raw_pagc_tics = np.array(raw_pagc_tics)
        meta_predictors = np.array(meta_predictors)
        # scale meta_data, scale raw_data
        if self.scale_raw:
            self.raw_scaler.scale_mz(raw_predictors, centers)
            self.raw_scaler.scale_intens(raw_predictors, raw_pagc_tics)
            self.raw_scaler.scale_intens(raw_predictors,raw_pagc_filltime)

        if self.scale_meta:
            meta_train = meta_predictors[border1s[0] : border2s[0]]
            self.meta_scaler.fit(meta_train)
            meta_data = self.meta_scaler.transform(meta_predictors)

        self.data_x_meta = meta_data[border1:border2]
        self.data_x_raw = raw_predictors[border1:border2]
        self.data_y = predictions[border1:border2]


    def __getitem__(self, index):
        # self.data_x_meta
        # self.data_x_raw
        # self.data_y

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_raw = self.data_x_raw[s_begin:s_end]
        seq_x_meta = self.data_x_meta[s_begin:s_end]
        seq_x_raw = np.concatenate( # batch, timestep, 2, maxlen
            [seq_x_raw, np.zeros((self.pred_len, self.data_x_raw.shape[1], self.data_x_raw[2], self.data_x_raw[3]))], axis=0
        )
        seq_x_meta = np.concatenate(
            [seq_x_meta, np.zeros((self.pred_len, self.data_x_meta.shape[-1]))], axis=0
        )


        seq_y = self.data_y[s_end:r_end]
        mask = np.concatenate([np.zeros(self.seq_len), np.zeros(self.pred_len)], axis=0)
        mask = mask[:, None]

        # Add the mask to the timestamps: # 480, 5
        # mark = np.concatenate([mark, mask[:, np.newaxis]], axis=1)

        seq_x_raw = seq_x_raw.astype(np.float32)
        seq_x_meta = seq_x_meta.astype(np.float32)
        seq_y = seq_y.astype(np.float32)
        mask = mask.astype(np.int64)

        return torch.tensor(seq_x_raw), torch.tensor(seq_y), torch.tensor(seq_x_meta), torch.tensor(mask)

    def __len__(self):
        return len(self.data_x_meta) - self.seq_len - self.pred_len + 1


    @property
    def d_input(self):
        return self.data_x_meta.shape[-1]

    @property
    def d_output(self):
        if self.features in ["M", "S"]:
            return self.data_x_meta.shape[-1]
        elif self.features == "MS":
            return 1
        else:
            raise NotImplementedError

    @property
    def n_tokens_time(self):
        if self.freq == 'h':
            return [13, 32, 7, 24]
        elif self.freq == 't':
            return [13, 32, 7, 24, 4]
        else:
            raise NotImplementedError


class _Dataset_AGC(MassSpecDataset):
    def __init__(self, data_path="WTH.csv", target="filltime", **kwargs):
        super().__init__(data_path=data_path, target=target, **kwargs)


class AGC(MassSpecSequenceDataset):
    _name_ = "agc"

    _dataset_cls = _Dataset_AGC

    init_defaults = {
        "size": None,
        "offset": 4,
        "target": "filltime",
        "variant": 0,
        "scale_raw": True,
        "scale_meta": True,
        "maxIT": 500,
        "meta_features": True,
        "timestep": 5,
    }

    variants = {
        0: "BSA_pAGC_training.arrow",
        1: "BSA_pAGC_training_2.arrow",
        2: "BSA_pAGC_training_3.arrow",
        3: "HELA_pAGC_training.arrow",
        4: "HELA_pAGC_training_2.arrow",
        5: "HELA_pAGC_training_3.arrow",
        6: "HELA_pAGC_training_4.arrow",
        7: "HELA_pAGC_training_5.arrow",
        8: "HELA_pAGC_training_6.arrow",
        9: "HELA_pAGC_training_7.arrow",
        10: "HELA_pAGC_training_8.arrow",
        11: "HELA_pAGC_training_9.arrow",
        12: "HELA_pAGC_training_10.arrow",
        13: "HELA_pAGC_training_15_under400width.arrow"

    }
