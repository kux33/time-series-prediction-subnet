# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshi
# Copyright © 2023 Taoshi, LLC

import os
import uuid
import random
import time
from typing import List, Tuple

import numpy as np

from data_generator.binance_data import BinanceData
from template.protocol import Forward
from time_util.time_util import TimeUtil
from vali_objects.cmw.cmw_objects.cmw_client import CMWClient
from vali_objects.cmw.cmw_objects.cmw_miner import CMWMiner
from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_config import ValiConfig
from vali_objects.dataclasses.prediction_data_file import PredictionDataFile
from vali_objects.scaling.scaling import Scaling
from vali_objects.scoring.scoring import Scoring

import bittensor as bt


if __name__ == "__main__":
    # Testing everything locally outside of the bittensor infra to ensure logic works properly

    client_request = ClientRequest(
        client_uuid=str(uuid.uuid4()),
        stream_type="BTCUSDT",
        topic_id=1,
        schema_id=1,
        feature_ids=[0.001, 0.002, 0.003, 0.004],
        prediction_size=int(random.uniform(ValiConfig.PREDICTIONS_MIN, ValiConfig.PREDICTIONS_MAX))
    )

    start_dt, end_dt, ts_ranges = ValiUtils.randomize_days(True)
    print("start", start_dt)
    print("end", end_dt)

    data_structure = [[], [], [], []]

    for ts_range in ts_ranges:
        BinanceData.get_data_and_structure_data_points(client_request.stream_type, data_structure, ts_range)

    vmins, vmaxs, dp_decimal_places, scaled_data_structure = Scaling.scale_data_structure(data_structure)

    request_uuid = str(uuid.uuid4())
    test_vali_hotkey = str(uuid.uuid4())
    # should be a hash of the vali hotkey & stream type (1 is fine)
    stream_id = hash(str(client_request.stream_type) + test_vali_hotkey)
    # close, high, low, volume
    samples = bt.tensor(scaled_data_structure)

    forward_proto = Forward(
        request_uuid=request_uuid,
        stream_id=stream_id,
        samples=samples,
        topic_id=client_request.topic_id,
        feature_ids=client_request.feature_ids,
        schema_id=client_request.schema_id,
        prediction_size=client_request.prediction_size
    )

    vm = ValiUtils.get_vali_records()
    client = vm.get_client(client_request.client_uuid)
    if client is None:
        print("client doesnt exist")
        cmw_client = CMWClient().set_client_uuid(client_request.client_uuid)
        cmw_client.add_stream(CMWStreamType().set_stream_id(stream_id).set_topic_id(client_request.topic_id))
        vm.add_client(cmw_client)
    else:
        client_stream_type = client.stream_exists(stream_id)
        if client_stream_type is None:
            client.add_stream(CMWStreamType().set_stream_id(stream_id).set_topic_id(client_request.topic_id))
    print(CMWUtil.dump_cmw(vm))
    ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(vm))

    # feel free to override for live data
    # preds = 2

    print("number of preds", client_request.prediction_size)
    for a in range(0, 25):
        s_predictions = np.array([random.uniform(0.499, 0.501) for i in range(0, client_request.prediction_size)])

        #using proto for testing
        forward_proto.predictions = bt.tensor(s_predictions)
        # convert to millis
        ts = TimeUtil.minute_in_millis(client_request.prediction_size * 5)

        # unscaled_data_structure = Scaling.unscale_values(vmins[0],
        #                                                  vmaxs[0],
        #                                                  dp_decimal_places[0],
        #                                                  forward_proto.predictions.numpy())
        output_uuid = str(uuid.uuid4())
        miner_uuid = str(uuid.uuid4())
        # just the vali hotkey for now

        pdf = PredictionDataFile(
            client_uuid=client_request.client_uuid,
            stream_type=client_request.stream_type,
            stream_id=stream_id,
            topic_id=client_request.topic_id,
            request_uuid=request_uuid,
            miner_uid=miner_uuid,
            start=TimeUtil.timestamp_to_millis(end_dt),
            end=TimeUtil.timestamp_to_millis(end_dt)+ts,
            vmins=vmins,
            vmaxs=vmaxs,
            decimal_places=dp_decimal_places,
            predictions=forward_proto.predictions.numpy()
        )
        ValiUtils.save_predictions_request(output_uuid, pdf)

    print("remove stale files")
    ValiBkpUtils.delete_stale_files(ValiBkpUtils.get_vali_predictions_dir())

    predictions_to_complete = []

    while True:
        break_now = False
        predictions_to_complete.extend(ValiUtils.get_predictions_to_complete())
        # for file in all_files:
        #     unpickled_data_file = ValiUtils.get_vali_predictions(file)
        #     if TimeUtil.now_in_millis() > unpickled_data_file.end:
        #         unpickled_unscaled_data_structure = Scaling.unscale_values(unpickled_data_file.vmins[0],
        #                                                                    unpickled_data_file.vmaxs[0],
        #                                                                    unpickled_data_file.decimal_places[0],
        #                                                                    unpickled_data_file.predictions)
        #         if unpickled_data_file.request_uuid not in request_to_complete:
        #             request_to_complete[unpickled_data_file.request_uuid] = {
        #                 "po": unpickled_data_file,
        #                 "predictions": {}
        #             }
        #         request_to_complete[
        #             unpickled_data_file.request_uuid][
        #             "predictions"][unpickled_data_file.miner_uid] = unpickled_unscaled_data_structure
        #         os.remove(file)
        if len(predictions_to_complete) > 0:
            break
        print("going back to sleep to wait...")
        time.sleep(60)

    print("request time done!")
    # print(request_to_complete)

    updated_vm = ValiUtils.get_vali_records()

    for request_details in predictions_to_complete:
        request_df = request_details.df
        data_structure = [[], [], [], []]
        BinanceData.get_data_and_structure_data_points(request_df.stream_type,
                                                       data_structure,
                                                       (request_df.start, request_df.end))
        print("results:", data_structure[0])
        scores = {}
        for miner_uid, miner_preds in request_details.predictions.items():
            scores[miner_uid] = Scoring.score_response(miner_preds, data_structure[0])

        scores["test"] = Scoring.score_response([row*0.999 for row in data_structure[0]], data_structure[0])
        scores["test1"] = Scoring.score_response([row*0.9975 for row in data_structure[0]], data_structure[0])
        scores["test2"] = Scoring.score_response([row*0.995 for row in data_structure[0]], data_structure[0])
        scores["test3"] = Scoring.score_response([row*0.9925 for row in data_structure[0]], data_structure[0])
        scores["test4"] = Scoring.score_response([row * 0.99 for row in data_structure[0]], data_structure[0])

        scaled_scores = Scoring.scale_scores(scores)
        print("client_uuid", request_df.client_uuid)
        print("stream type", request_df.stream_id)
        print("request_uuid", request_details.request_uuid)
        stream_id = updated_vm.get_client(request_df.client_uuid).get_stream(request_df.stream_id)

        # add scaled scores
        for miner_uid, scaled_score in scaled_scores.items():
            stream_miner = stream_id.get_miner(miner_uid)
            if stream_miner is None:
                stream_miner = CMWMiner(miner_uid, 0, 0, [])
                stream_id.add_miner(stream_miner)
            stream_miner.add_score(scaled_score)

        sorted_scores = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
        top_winners = sorted_scores[:10]
        top_winner_miner_uids = [item[0] for item in top_winners]

        weighed_scores = Scoring.weigh_miner_scores(top_winners)

        print("winning scores", top_winners)
        print("winning distribution", weighed_scores)

        for file in request_details.files:
            os.remove(file)

    print(CMWUtil.dump_cmw(updated_vm))
    ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(updated_vm))
