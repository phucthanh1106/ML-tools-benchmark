from clearml import Dataset
try:
    dataset = Dataset.get(dataset_id='450ffa4202694729abe163c2b843fef7')
    print("Your raw dataset path is:", dataset.get_local_copy())
except Exception as e:
    print("Error pulling from ClearML:", e)