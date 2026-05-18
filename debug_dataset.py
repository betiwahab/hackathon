from dataset import MedicalDataset

d = MedicalDataset(csv_file='data/ABreast-Classification.csv', image_dir='data/images')
print('len', len(d))
image, label, patient_id, image_id = d[0]
print('sample', patient_id, image_id, label.item())
