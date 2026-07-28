
'''
Module to keep track of the usage of the different features of the library. 
Asks nicely for permission before saving anything.

'''
import os as _os
import atexit

from enum import Enum

usage_categories = Enum('uses', ['FuzzySets', 'Funcs', 'Persistence', 'Visualization', 'FuzzyCognitiveMaps', 'RuleMining', 'Classification'])

def instance_dict_usage():
    # Inits the categories to track
    return {usage_categories.FuzzySets: {'t1':0 , 't2': 0, 'gt2': 0, 'temporal': 0, 'temporal_t2': 0, 'categorical': 0},
                usage_categories.Funcs: {'fit': 0, 'precompute_labels': 0, 'opt_labels':0},
                usage_categories.Persistence : {'persistence_write': 0,
                    'persistence_read': 0},
                usage_categories.FuzzyCognitiveMaps : {'fcm_create': 0, 'fcm_report': 0},
                usage_categories.Visualization : {'plot_rules': 0, 'plot_graph': 0, 'print_rules': 0, 'plot_fuzzy_variable': 0},
                usage_categories.RuleMining : {'mine_rulebase' : 0},
                usage_categories.Classification : {'double_go': 0, 'data_mining': 0}
    }


def rename_dict_usage_categories():
    global usage_data
    str_names = ['Fuzzy sets', 'Utils and training', 'Persistence', 'Fuzzy Cognitive Maps', 'Visualization', 'Rule Mining', 'Classification']
    new_usage = {}

    for ix, (key, value) in enumerate(usage_data.items()):
        new_usage[str_names[ix]] = value

    usage_data = new_usage


def send_data():
    global path_usage_data_folder
    # Send the data to the developers using a sfpt server (TODO)
    import paramiko

    host = "sftp.upv.es"
    port = 22

    files_to_send = _os.listdir(path_usage_data_folder)

    # create ssh client 
    ssh_client = paramiko.SSHClient()

    # remote server credentials
    host = "hostname"
    # username = "username"
    # password = "password"
    port = port

    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=host,port=port)

    # create an SFTP client object
    ftp = ssh_client.open_sftp()

    for file in files_to_send:
        # send a file from the remote server
        ftp.put(path_usage_data_folder + '/' + file, '/home/' + file)
    
    # Send also the logger file
    ftp.put('ex_fuzzy.log', '/home/ex_fuzzy.log')
    
    # close the connection
    ftp.close()
    ssh_client.close()

    # Clear the logger file
    with open('ex_fuzzy.log', 'w') as f:
        f.write('')


def main():
    global path_usage_data_folder
    # Open the usage file from user
    file_name = 'usage_data_permission.txt'
    directory = _os.path.dirname(_os.path.realpath(__file__))

    if file_name not in _os.listdir(directory):
        # Get input from user
        print('exFuzzy:')
        print('Do you want to share your usage data with the developers?')
        print('This data will be used to improve the library. It will also help us to acquire funding for the project.')
        print('If you agree, we will store the number of times you use each functionality in a file called "usage_data.txt".')
        print('This file will be stored in the same folder as the library.')
        print('If you do not agree, we will not store any data.')
        print('If you agree, type "[y]es". If you do not agree, type "no".')
        user_input = None

        while user_input not in ['no', 'yes', 'y']:
            user_input = input()

            if user_input not in ['no', 'yes', 'y']:
                print('Answer not understood, please repeat: If you agree, type "[y]es". If you do not agree, type "no".')
        
        with open(_os.path.join(directory, file_name), 'w') as f:
            f.write(user_input)

    # Read the usage file
    with open(_os.path.join(directory, file_name), 'r') as f:
        user_input = f.read()

    save_usage_flag = user_input == 'yes' or user_input == 'y'

    if save_usage_flag:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(_os.path.join(directory, 'ex_fuzzy.log'))
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.DEBUG)
        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        # Create usage reports
        @atexit.register
        def save_usage():
            # Saves the usage in the corresponding file
            path_usage_data = _os.path.join(path_usage_data_folder, 'usage_data_' + str(number_files) + '.txt')
            rename_dict_usage_categories()
            import json
            # Save the usage as a json
            with open(_os.path.join(directory, path_usage_data), 'w') as f:
                json.dump(usage_data, f)
            
            # Send the data to the developers
            # send_data()
        # Create usage data dictionary
        usage_data = instance_dict_usage()

        path_usage_data_folder = 'usage_data'
        if path_usage_data_folder not in _os.listdir(directory):
            _os.mkdir(_os.path.join(directory, path_usage_data_folder))
        
        number_files = len(_os.listdir(_os.path.join(directory, path_usage_data_folder)))

        if number_files > 100:
            # Avoid a large number of files
            while number_files > 100:
                _os.remove(_os.path.join(directory, path_usage_data_folder, 'usage_data_' + str(number_files-1) + '.txt'))
                number_files = len(_os.listdir(path_usage_data_folder))


        atexit.register(save_usage)

save_usage_flag = False
if __name__ == '__main__':
    pass