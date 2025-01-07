import os 
import json


def merge(source_folder , output_folder, filename):
    # source_folder = path + '/query/query_json/'
    # output_folder = path + '/query/queries_merged'
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)  

    save_path = os.path.join(output_folder, filename)      
    json_files = [pos_json for pos_json in os.listdir(source_folder) if pos_json.endswith('.json')]
    data = []
    for json_file in json_files:
        print ('===============',os.path.join(source_folder, json_file))
        with open(os.path.join(source_folder, json_file)) as f:
            tmp = json.load(f)
            if isinstance(tmp, list) and len(tmp) == 1:
                tmp = tmp[0]
            if isinstance(tmp, list):
                data += tmp
            else:
                data.append(tmp)
    length = len(data)

    if os.path.exists(output_folder):
        pass
    else:
        os.makedirs(output_folder)

    with open(save_path, 'w') as f:
        json.dump(data, f)
    print(f"Successfully merged {length} json files")