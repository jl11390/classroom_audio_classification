def get_label_dict(target_class_version):
    assert target_class_version in [0, 1], "target class is invalid"
    if target_class_version == 0:  # full target class
        label_dict = {
            'Lecturing': 0,
            'Q/A': 1,
            'Teacher-led Conversation': 2,
            'Student Presentation': 3,
            'Individual Student Work': 4,
            'Collaborative Student Work': 5,
            'Other': 6
        }
    if target_class_version == 1:  # reduced target class
        label_dict = {
            'Lecturing': 0,
            'Q/A': 0,
            'Teacher-led Conversation': 0,
            'Student Presentation': 0,
            'Individual Student Work': 1,
            'Collaborative Student Work': 2,
            'Other': 3
        }
    
    label_dict_lower = {}
    label_dict_upper = {}
    
    for target, target_class in label_dict.items():
        if target != target.lower():
            label_dict_lower[target.lower()] = target_class
        if target != target.upper():
            label_dict_upper[target.upper()] = target_class
    label_dict.update(label_dict_lower)
    label_dict.update(label_dict_upper)

    return label_dict

def get_reverse_label_dict(target_class_version):
    assert target_class_version in [0, 1], "target class is invalid"
    if target_class_version == 0:  # full target class
        label_dict = {
            'Lecturing': 0,
            'Q/A': 1,
            'Teacher-led Conversation': 2,
            'Student Presentation': 3,
            'Individual Student Work': 4,
            'Collaborative Student Work': 5,
            'Other': 6
        }
    if target_class_version == 1:  # reduced target class
        label_dict = {
            'Lecturing': 0,
            'Q/A': 0,
            'Teacher-led Conversation': 0,
            'Student Presentation': 0,
            'Individual Student Work': 1,
            'Collaborative Student Work': 2,
            'Other': 3
        }
    
    keys = list(label_dict.keys())
    values = list(label_dict.values())
    reverse_label_dict = {}
    for value, key in zip(values, keys):
        if value in reverse_label_dict:
            reverse_label_dict[value] = reverse_label_dict[value] + ' & ' + key
        else:
            reverse_label_dict[value] = key

    return reverse_label_dict

if __name__ == "__main__":
    label_dict = get_label_dict(0)
    print(label_dict)
    reverse_label_dict = get_reverse_label_dict(0)
    print(reverse_label_dict)