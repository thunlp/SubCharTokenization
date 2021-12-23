import json
import pickle
import os
from mrc.preprocess.CHID_preprocess import RawResult, logits_matrix_to_array

seeds = [2, 23, 234]


def get_final_predictions(all_results, tmp_predict_file, g=True):
    # if not os.path.exists(tmp_predict_file):
    #     pickle.dump(all_results, open(tmp_predict_file, 'wb'))

    raw_results = {}
    for i, elem in enumerate(all_results):
        example_id = elem.example_id
        if example_id not in raw_results:
            raw_results[example_id] = [(elem.tag, elem.logit)]
        else:
            raw_results[example_id].append((elem.tag, elem.logit))

    results = []
    for example_id, elem in raw_results.items():
        index_2_idiom = {index: tag for index, (tag, logit) in enumerate(elem)}
        logits = [logit for _, logit in elem]
        if g:
            results.extend(logits_matrix_to_array(logits, index_2_idiom))
        else:
            results.extend(logits_matrix_max_array(logits, index_2_idiom))
    return results



def write_predictions(results, output_prediction_file):
    # output_prediction_file = result6.csv
    # results = pd.DataFrame(results)
    # results.to_csv(output_prediction_file, header=None, index=None)

    results_dict = {}
    for result in results:
        results_dict[result[0]] = result[1]
    with open(output_prediction_file, 'w') as w:
        json.dump(results_dict, w, indent=2)

    print("Writing predictions to: {}".format(output_prediction_file))

for seed in seeds:
    tmp_predict_file = 'logs/chid/raw_zh/' + str(seed) +'/raw_predictions.pkl'
    json_file = 'logs/chid/raw_zh/' + str(seed) + '/test_predictions.json'
    result = pickle.load(open(tmp_predict_file, 'rb'))
    results = get_final_predictions(result, tmp_predict_file, g=True)
    write_predictions(results, json_file)