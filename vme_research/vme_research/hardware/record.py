"""Record and Load classes"""

from __future__ import annotations

from datetime import datetime
import glob

import os
import json
import numpy as np
import pandas as pd
import pickle
from vme_research.messaging.shared_ndarray import SharedNDArray


def make_sequence_directory(dataset_prefix):
    date_str = datetime.today().strftime("%Y_%m_%d")
    dataset_dir = dataset_prefix + "_" + date_str + "_data"

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    previous_recordings = sorted(glob.glob(os.path.join(dataset_dir, "sequence_*")))

    if len(previous_recordings) > 0:
        last_recording_id = int(previous_recordings[-1][-6:])
        recording_id = last_recording_id + 1
    else:
        recording_id = 0

    sequence_dir = os.path.join(dataset_dir, "sequence_{:06d}".format(recording_id))
    os.makedirs(sequence_dir)

    return sequence_dir


def get_latest_sequence_directory(dataset_prefix):
    previous_dataset_dirs = sorted(glob.glob(dataset_prefix + "_*_data"))
    lastest_dataset_dir = previous_dataset_dirs[-1]
    previous_recordings = sorted(
        glob.glob(os.path.join(lastest_dataset_dir, "sequence_*"))
    )
    return previous_recordings[-1]


class Record:
    def __init__(self, save_directory, time_source=None, fields_options=None):
        self.save_directory = save_directory
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        if fields_options is not None:
            self.set_fields_options(fields_options)

        self.time_source = time_source

    def set_fields_options(self, fields_options):
        self.fields_options = fields_options
        self.data_lists = [[] for options in fields_options["fields"]]
        self.t_received_list = []
        self.t_list = []

        for options in self.fields_options["fields"]:
            # Make directory to hold npy files if it does not exist
            if options["split"]:
                field_directory = os.path.join(self.save_directory, options["name"])
                if not os.path.exists(field_directory):
                    os.makedirs(field_directory)

    def pub(self, t, data, t_received=None):
        if t_received is None and self.time_source is not None:
            self.t_received_list.append(self.time_source.time())
        else:
            self.t_received_list.append(t_received)
        self.t_list.append(t)
        assert len(self.t_received_list) == len(self.t_list)

        for options, data_list, d in zip(
            self.fields_options["fields"], self.data_lists, data
        ):
            if options["split"]:
                relative_path = os.path.join(
                    options["name"],
                    "{}_{:08d}.npy".format(options["name"], len(self.t_list) - 1),
                )
                data_list.append(relative_path)

                full_path = os.path.join(self.save_directory, relative_path)
                if options["type"] == str(np.ndarray):
                    if type(d) == SharedNDArray:
                        d = d.x
                    np.save(full_path, d)
                else:
                    with open(full_path, "wb") as handle:
                        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                if type(d) == SharedNDArray:
                    d = np.copy(d.x)
                data_list.append(d)

    def close(self, append_values=None):
        info_dict = {}
        info_dict["fields_options"] = self.fields_options
        info_dict["fields"] = {}

        np.save(os.path.join(self.save_directory, "t"), np.array(self.t_list))
        info_dict["fields"]["t"] = "t.npy"

        np.save(
            os.path.join(self.save_directory, "t_received.npy"),
            np.array(self.t_received_list),
        )
        info_dict["fields"]["t_received"] = "t_received.npy"

        field_type_tuples = []
        if append_values:
            info_dict["append_fields"] = {}
            field_type_tuples.append(
                [
                    "append_fields",
                    info_dict["fields_options"]["append_fields"],
                    append_values,
                ]
            )
        field_type_tuples.append(
            ("fields", info_dict["fields_options"]["fields"], self.data_lists)
        )

        for fields_type, fields_options, field_values in field_type_tuples:
            for options, d in zip(fields_options, field_values):
                if options["type"] == str(np.ndarray) and not options["split"]:
                    npy_name = options["name"] + ".npy"
                    np.save(os.path.join(self.save_directory, npy_name), np.array(d))
                    info_dict[fields_type][options["name"]] = npy_name
                else:
                    info_dict[fields_type][options["name"]] = d

        with open(
            os.path.join(self.save_directory, "data.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(info_dict, f, ensure_ascii=False, indent=2)


class Load:
    def __init__(self, save_directory, tskip=None):
        self.save_directory = save_directory
        self.tskip = tskip
        self.t_shift = 0.0
        self.t_received_shift = 0.0

        with open(
            os.path.join(self.save_directory, "data.json"), "r", encoding="utf-8"
        ) as f:
            self.info_dict = json.load(f)

        if "fields" in self.info_dict:
            self.info_dict["fields"]["t"] = np.load(
                os.path.join(self.save_directory, self.info_dict["fields"]["t"])
            )
            self.info_dict["fields"]["t_received"] = np.load(
                os.path.join(
                    self.save_directory, self.info_dict["fields"]["t_received"]
                )
            )

        for field_type in ["fields", "append_fields"]:
            for options in self.info_dict["fields_options"][field_type]:
                if options["type"] == str(np.ndarray) and not options["split"]:
                    npy_name = self.info_dict[field_type][options["name"]]
                    self.info_dict[field_type][options["name"]] = np.load(
                        os.path.join(self.save_directory, npy_name)
                    )

        if self.tskip is not None:
            self.last_t_index = np.searchsorted(
                self.info_dict["fields"]["t_received"], self.tskip
            )
        else:
            self.last_t_index = 0

    def _load_field_at_time(self, t_index):
        fields_data = []
        for options in self.info_dict["fields_options"]["fields"]:
            # load as a numpy array if it is split into multiple numpy arrays?
            if options["type"] == str(np.ndarray) and options["split"]:
                # print(self.last_t_index, t_array.shape[0], len(self.info_dict['fields'][options['name']]))
                npy_array = np.load(
                    os.path.join(
                        self.save_directory,
                        self.info_dict["fields"][options["name"]][t_index],
                    )
                )
                fields_data.append(npy_array)
            else:
                fields_data.append(self.info_dict["fields"][options["name"]][t_index])
        return fields_data

    def get(
        self, t, ret_t_received=False
    ):  # -> tuple[bool, float | None, list | tuple]: TODO: levi does not know how to update the type annotations after adding ret_t_received
        t_array = self.info_dict["fields"]["t_received"]
        assert t_array.shape[0] == self.info_dict["fields"]["t"].shape[0]
        if self.last_t_index < t_array.shape[0] and t_array[self.last_t_index] < t:

            if ret_t_received:
                t_received = self.info_dict["fields"]["t_received"][self.last_t_index]
            t_return = self.info_dict["fields"]["t"][self.last_t_index]
            fields_data = self._load_field_at_time(self.last_t_index)

            self.last_t_index += 1

            if ret_t_received:
                return True, t_received, t_return, fields_data
            else:
                return True, t_return, fields_data
        else:
            if ret_t_received:
                return False, None, None, (None,)
            else:
                return False, None, (None,)

    def fields(self):
        # iterate over the fields at each time step
        for t_index in range(0, self.info_dict["fields"]["t_received"].shape[0]):
            t_received = self.info_dict["fields"]["t_received"][t_index]
            t_return = self.info_dict["fields"]["t"][t_index]
            fields_data = self._load_field_at_time(t_index)
            yield t_return, t_received, fields_data

    def get_all(self) -> dict:
        return self.info_dict["fields"]

    def get_appended(self):
        return self.info_dict["append_fields"]

    def apply_tshift(self, t_received_shift, t_shift):
        self.t_received_shift = t_received_shift
        self.t_shift = t_shift

        self.get_all()["t_received"] = (
            self.get_all()["t_received"] + self.t_received_shift
        )
        self.get_all()["t"] = self.get_all()["t"] + self.t_shift

    @classmethod
    def time_synchronization(cls, loader_1: Load, *loaders: Load):
        """Takes in an initial loader that will be used to set true 0, and an
        unspecified amount of loader arguments that will be time synchronized to that initial loader
        """
        load_1_data = loader_1.get_all()
        loader_1.apply_tshift(
            t_received_shift=-load_1_data["t_received"][0], t_shift=-load_1_data["t"][0]
        )

        for loader in loaders:
            load_2_data = loader.get_all()
            loader.apply_tshift(
                t_received_shift=loader_1.t_received_shift,
                t_shift=-load_2_data["t"][0]
                + load_2_data["t_received"][0]
                + loader_1.t_received_shift,
            )


# A specialized loader class for EventStreams
# TODO mmap instead of loading into ram
class LoadEventStream(Load):
    def __init__(self, save_directory, tskip=None):
        super().__init__(save_directory, tskip)
        # TODO Hack in place load
        print(os.path.join(self.save_directory, self.get_appended()["events_t"]))
        self.get_appended()["events_t"] = np.load(os.path.join(self.save_directory, self.get_appended()["events_t"])).astype(np.int64)
        self.get_appended()["events_xy"] = np.load(os.path.join(self.save_directory, self.get_appended()["events_xy"]))
        self.get_appended()["events_p"] = np.load(os.path.join(self.save_directory, self.get_appended()["events_p"]))

    def apply_tshift(self, t_received_shift, t_shift):
        super().apply_tshift(t_received_shift, t_shift)
        self.get_appended()["events_t"] += int(self.t_shift * 1000000)

class LoadPd(Load):
    # NOTE:
    def __init__(self, save_directory, tskip=None):
        self.save_directory = save_directory
        self.tskip = tskip

        data = pd.DataFrame()
        with open(
            os.path.join(self.save_directory, "data.json"), "r", encoding="utf-8"
        ) as f:
            self.info_dict = json.load(f)

        if "fields" in self.info_dict:
            data["t"] = np.load(
                os.path.join(self.save_directory, self.info_dict["fields"]["t"])
            )
            data["t_received"] = np.load(
                os.path.join(
                    self.save_directory, self.info_dict["fields"]["t_received"]
                )
            )

        for field_type in ["fields", "append_fields"]:
            for options in self.info_dict["fields_options"][field_type]:
                if options["type"] == str(np.ndarray) and not options["split"]:
                    npy_name = self.info_dict[field_type][options["name"]]
                    data[options["labels"]] = np.load(
                        os.path.join(self.save_directory, npy_name)
                    )
                if options["split"]:

                    data[options["labels"]] = self.info_dict["fields"][options["name"]]

        if self.tskip is not None:
            self.last_t_index = np.searchsorted(data["t_received"], self.tskip)
        else:
            self.last_t_index = 0

        self.info_dict["fields"] = data

    def _load_field_at_time(self, t_index):
        fields_data = []
        # TODO: See if we can make this pandas also
        for options in self.info_dict["fields_options"]["fields"]:

            # load as a numpy array if it is split into multiple numpy arrays?
            if options["type"] == str(np.ndarray) and options["split"]:
                # print(self.last_t_index, t_array.shape[0], len(self.info_dict['fields'][options['name']]))
                npy_array = np.load(
                    os.path.join(
                        self.save_directory,
                        self.info_dict["fields"][options["labels"]][t_index],
                    )
                )
                fields_data.append(npy_array)
            else:
                fields_data.append(
                    self.info_dict["fields"].iloc[t_index][options["labels"]]
                )
        return fields_data

    def get_all(self) -> pd.DataFrame:
        return super().get_all()  # type: ignore
