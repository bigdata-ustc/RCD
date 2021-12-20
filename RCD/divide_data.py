import json
import random


min_log = 0 #15


def divide_data():
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set and test_set (0.8:0.2)
    :return:
    '''
    with open('../data/junyi/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    # 1. delete students who have fewer than min_log response logs
    stu_i = 0
    l_log = 0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < min_log:
            del stus[stu_i]
            stu_i -= 1
        else:
            l_log += stus[stu_i]['log_num']
        stu_i += 1
    # 2. divide dataset into train_set and test_set
    train_set, test_set = [], []
    for stu in stus:
        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * 0.8)
        test_size = stu['log_num'] - train_size
        logs = []
        for log in stu['logs']:
            logs.append(log)
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[train_size:]
        test_set.append(stu_test)
        # shuffle logs in train_slice together, get train_set
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})
    random.shuffle(train_set)
    with open('../data/junyi/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('../data/junyi/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set

if __name__ == '__main__':
    divide_data()
