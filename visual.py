import pandas as pd
import cv2

# Read the CSV file

import pandas as pd
import cv2
import os

# Read the CSV file
df = pd.read_csv('/data/ephemeral/baseline/ultralytics-main/runs/detect/divide/stratified_class_1_m_focal8/submission_stratified_class_1_m_focal8.csv')

# Ensure the output directory exists
output_dir = 'path_to_output'
os.makedirs(output_dir, exist_ok=True)

# Loop through the dataframe
for index, row in df.iterrows():
    image_id = row['image_id']
    prediction_string = row['PredictionString']

    # Load image
    image_path = f'/data/ephemer/dataset/{image_id}.jpg'
    image = cv2.imread(image_path)

    if image is not None:
        # Parse the PredictionString
        predictions = prediction_string.split()
        for i in range(0, len(predictions), 6):
            label, score, xmin, ymin, xmax, ymax = predictions[i:i+6]
            score = float(score)
            # Draw a box only if the score is greater than 0.05
            if score > 0.05: break
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(image, f'{label}: {score:.2f}', (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save the annotated image
        cv2.imwrite(os.path.join(output_dir, f'{image_id}_annotated.jpg'), image)

# Print a message when done
print("All images have been processed and saved.")