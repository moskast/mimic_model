# mimic-lstm

This is a complete preprocessing, model training, and figure generation repo for an adapted version of "An attention based deep learning model of clinical events in the intensive care unit".
It allows the use of MIMIC-III and MIMIC-IV and produces PyTorch models.

### Getting Started

Put all MIMIC CSVs into the correct "data/mimic_X_database" folder.
The program will try to create the rest of the folders itself.

The pipeline is split into three parts:
- The first is parsing the mimic data set into a single file (performed by mimic_parser.py)
- The second is generating data set for each target out of the parsed file (performed by mimic_pre_processor.py)
- The third is training the models (performed by trained_model.py)

Settings of which part to perform as well as the mimic version can be changed in train.py or
by directly calling main(parse_mimic, pre_process_data, create_models, mimic_version=4).

Models and figures are generated in the test.ipynb notebook.
Simply adjusting the target to 'MI', 'Sepsis', or 'Vancomycin' will generate the figures panels and images required for each part of the figure.

### Prerequisites
Refer to env.yml

### License
This project is licensed under the MIT License - see the LICENSE.md file for details

### Acknowledgments
I thank Deepak A. Kaji for laying the groundwork of this implementation in "An attention based deep learning model of clinical events in the intensive care unit"