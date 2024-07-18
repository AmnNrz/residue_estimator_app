import tensorflow as tf

# TensorFlow SavedModel path
saved_model_dir = './saved_models/exported_model'

# Convert the model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Step 3: Save the converted TFLite model to a file
tflite_model_file = './saved_models/exported_model.tflite'
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {tflite_model_file}")

# Load the TFLite model
tflite_model_file = './saved_models/exported_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)

# Allocate tensors
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details
print("Input details:")
for input_detail in input_details:
    print(f"Name: {input_detail['name']}")
    print(f"Shape: {input_detail['shape']}")
    print(f"Data type: {input_detail['dtype']}")
    print("")

# Print output details
print("Output details:")
for output_detail in output_details:
    print(f"Name: {output_detail['name']}")
    print(f"Shape: {output_detail['shape']}")
    print(f"Data type: {output_detail['dtype']}")
    print("")