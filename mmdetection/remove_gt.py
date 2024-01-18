import json

def process_coco_dataset(json_file, output_path):
# Load JSON data
	with open(json_file, 'r') as file:
		data = json.load(file)

# This dictionary will hold the count of annotations per image
	annotation_count = {}
	zero_cnt = {}

# Counting the number of annotations per image
	for annotation in data['annotations']:
		image_id = annotation['image_id']
		annotation_count[image_id] = annotation_count.get(image_id, 0) + 1
		if not annotation['category_id']:
			zero_cnt[image_id] = zero_cnt.get(image_id, 0) + 1
		else:
			zero_cnt[image_id] = zero_cnt.get(image_id, 0)

	# Filter out annotations with category_id of 0 where the image has more than one annotation
	filtered_annotations = [annotation for annotation in data['annotations']
							if not (zero_cnt[annotation['image_id']] != annotation_count[annotation['image_id']] and annotation['category_id'] == 0)]

	# Updating the annotations in the data
	data['annotations'] = filtered_annotations

	# Save the updated data back to a new JSON file
	with open(output_path, 'w') as outfile:
		json.dump(data, outfile, indent=2, separators=(',', ': '))

	return json_file

# Example usage
json_path = '/data/ephemeral/home/data/dataset/val_333_fold_1.json'
output_path = '/data/ephemeral/home/data/dataset/val_333_fold_1_remove_gt.json'
process_coco_dataset(json_path, output_path)