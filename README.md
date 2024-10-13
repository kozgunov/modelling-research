#  🚢 Advanced Ship-Models Real-Time Object Detection & Race Analysis (YOLOv8n)

DeepSight is a machine learning project that leverages YOLOv8n, a powerful object detection model, for analyzing maritime environments. The goal of this project is to identify and classify objects such as buoys, vessel numbers, models with numbers, models without numbers, and trails left by vessels. This solution aims to enhance maritime safety by providing real-time insights using computer vision techniques.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [5-classes-detected](#5-classes-detected)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)


- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Introduction

DeepSight is a machine learning project that leverages YOLOv8n, a powerful object detection model, for analyzing maritime environments. The goal of this project is to identify and classify objects such as buoys, vessel numbers, models with numbers, models without numbers, and trails left by vessels. This solution aims to enhance maritime safety by providing real-time insights using computer vision techniques.

## Features

- **Object Detection**: Detects and classifies 5 different classes related to maritime objects.
- **Real-Time Analysis**: Capable of analyzing images in real-time for immediate feedback.
- **Training and Customization**: Includes training scripts for customizing the model with new data.
- **Deployment Ready**: Can be deployed using Flask, Docker, or cloud services for real-time inference.
- **Mobility Deployment**: Model can be deployed in the modile phone on in another.

## 🌟 Features

Our project is packed with exciting features:

- 🔍 **YOLOv8n Model**: State-of-the-art object detection for 6 classes of maritime objects
- 🖼️ **Data Augmentation**: Enhancing our training data for robust model performance
- 🏋️ **Advanced Training**: Hyperparameter optimization and early stopping for optimal performance
- 🖥️ **HPC Integration**: Parallel training and distributed data processing for faster development
- 🌐 **Flask API**: Deploy your model with ease and integrate it into any system
- 📹 **Multi-Camera Processing**: Analyze races from multiple angles simultaneously
- 🎥 **Live Viewing**: Watch the race unfold with real-time object detection
- 🔢 **Intelligent Counting**: Keep track of completed rounds for numbered ships
- 👁️ **Occlusion Handling**: Don't lose track of ships, even when they overlap

## 🏗️ Project Structure

Our project is organized into several key components:

- `main.py`: The heart of our system, coordinating all operations
- `models.py`: Home to our YOLOModel and EnsembleModel classes
- `training.py`: Where the magic of model training happens
- `data_augmentation.py`: Spicing up our training data
- `data_processing.py`: Keeping our data neat and tidy
- `data_loading.py`: Efficient data loading for training
- `image_processing.py`: Handling all things image-related
- `video_processor.py`: Tackling video inputs with ease
- `ship_tracker.py`: Keeping an eye on every ship in the race
- `round_counter.py`: The digital scorekeeper for our maritime races
- `result_generator.py`: Crafting beautiful and informative race summaries
- `deploy.py`: Launching our model into the world
- `HPC_integration.py`: Supercharging our training with high-performance computing

## 🚀 Getting Started

Ready to dive in? Here's how to get started:

1. Clone this repository:   ```
   git clone https://github.com/kozgunov/modelling-research.git   ```
2. Install the required packages:   ```
   pip install -r requirements.txt   ```

## 🎮 Usage

Our system is versatile and easy to use. Here are some common operations:

### 🔧 Data Preparation

Prepare your dataset with:

- resize it to 600x600 size
- transfer photos to .png

## 🧠 Transfer Learning

We're standing on the shoulders of giants! Our project utilizes transfer learning, starting with the pre-trained YOLOv8n model. This approach allows us to leverage knowledge from vast datasets and apply it to our specific maritime detection task.

## 🔧 Advanced Features

- **Gradient Accumulation**: Implemented for effective training with limited GPU memory.
- **Mixed Precision Training**: Utilizing FP16 for faster training without sacrificing accuracy.
- **Early Stopping**: Prevents overfitting and saves time by stopping training when performance plateaus.
- **Distributed Training**: Leverage multiple GPUs or nodes for faster training.
- **Occlusion Handling**: Sophisticated tracking to maintain object identity even when ships overlap.

## 📊 Results and Visualization

After processing a race, you'll find:
- A summary video with annotations
- A text file with race results
- A visualization of the race results

## 🤝 Contributing

We love contributions! If you have ideas for improvements or new features, please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- The YOLOv8 team for their incredible object detection model
- Our maritime partners for providing real-world racing footage
- The open-source community for their invaluable tools and libraries

Set sail with our Ship Detection and Race Analysis system today! 🚢🏁









## 5 Classes Detected

- Buoy
- Number
- Model with number
- Model without number
- Trail after the model

### Prerequisites

- **Python 3.8+**
- **PyTorch**: For implementing and training the LLM.
- **NumPy**: For numerical computations.
- **Multiprocessing**: To simulate parallel operations.
- **Scikit-learn**: For utility functions and metrics.
- **Matplotlib**: For plotting results.
- **Optuna**: For hyperparameter optimization.
- **Yolov8**: Pre-trained model for detecting & post-training.


### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/kozgunov/modelling-research.git
   cd decentralized-llm-blockchain
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Upgrade pip**

   ```bash
   pip install --upgrade pip
   ```

4. **Install Dependencies**

   Install all required libraries using `pip`.

   ```bash
   pip install -r requirements.txt
   ```

   **Contents of `requirements.txt`:**

   ```text
   torch>=1.7.0
   numpy>=1.18.0
   scikit-learn>=0.23.0
   matplotlib>=3.2.0
   optuna>=4.0.0
   pandas>=2.0.0
   pip>=24.2
   sentencepiece
   ```

5. **Verify Installation**

   ```bash
   python -c "import torch; import numpy; import pandas; import pathlib; import PIL; import sklearn; import matplotlib; import optuna; print('All libraries installed successfully.')"
   ```

## Usage

### Running the Simulation

Execute the main script to run the simulation:

```bash
python main.py
```

### Data Preparation

- **Dataset**: Gathering of dataset is done by myself and not published anywhere. it consists from 7'500 photos for post-training.
- **Data Partitioning**: later...
- **Preprocessing**: later...

To prepare the data:

**Load and Prepare Data**

   ```python
   from data_preparation import load_and_prepare_superglue
   train_data, validation_data, test_data = load_and_prepare_superglue()
   ```


### Logging and Monitoring

- **Logging**: The system logs important events, errors, and performance metrics. In process...
- **Monitoring**: Use the logging output to monitor training progress, consensus operations, and attack simulations. In process...

## Code Structure

- **`model.py`**: later...
- **`analyze.py`**:  later...
- **`process_images.py`**:  later...
- **`training.py`**:  later...

# Components Description

## Performance Metrics

The system evaluates the global model using multiple performance metrics:

- **Accuracy**: later...
- **Perplexity**: later...
- **F1 Score**: later...

## Cloud Deployment

Deploy the model using cloud services such as AWS, Azure, or Google Cloud for scalability and reliability. Recommended options include:

- **AWS EC2** for persistent servers.
- **AWS Lambda** for serverless, on-demand inference.

## Results and Evaluation

- **Model Performance**: Training is going on. Results will  appear later...
- **Scalability**: Analysis of scalability is under research...
- **Efficiency**: Analysis of time and space complexity is under analysis...
- **Hyperparameter Optimization**: Hyperparameters are analyzed and given with the best performance as default value.

Refer to the `results` directory for detailed performance metrics, logs, and plots.

## Contributing

Contributing

We welcome contributions! If you have suggestions or improvements, feel free to fork the repository and submit a pull request. Please make sure to document any new features or changes to maintain code clarity.

1. **Fork the Repository**

   Click on the "Fork" button at the top-right corner of this page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/kozgunov/modelling-research.git
   ```

3. **Create a Feature Branch**

   ```bash
   git checkout -b feature/proposed-feature
   ```

4. **Make Changes and Commit**

   ```bash
   git add .
   git commit -m "add massage"
   ```

5. **Push to Your Fork**

   ```bash
   git push origin feature/proposed-feature
   ```

6. **Create a Pull Request**

Open a pull request from your fork's feature branch to the main repository's `master` branch.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## References

Please refer to the `REFERENCES.md` file for a list of academic papers and resources that inspired this project.

---

## Contact

For questions, suggestions, or support, please contact reserveknik@gmail.com

## Acknowledgements

- **YOLOv8n**: This project uses the YOLOv8 model, an advanced object detection system developed by Ultralytics.
- **Optuna**: [Optuna GitHub Repository](https://github.com/optuna/optuna)
- **NLTK and Rouge**: Libraries used for natural language processing and evaluation metrics.







