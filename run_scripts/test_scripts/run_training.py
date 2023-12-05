from run_scripts.test_scripts.train_with_data import run_train_process_with_data

sensors = {'empatica-left': 0,
           'empatica-right': 1,
           'bluesense-LUA': 2,
           'bluesense-LWR': 3,
           'bluesense-RUA': 4,
           'bluesense-RWR1': 5,
           'bluesense-RWR2': 6,
           'bluesense-TRS': 7,
           'maxim-red': 8,
           'maxim-green': 9, }

run_train_process_with_data(sensors['bluesense-RWR1'])
run_train_process_with_data(sensors['bluesense-RWR2'])
