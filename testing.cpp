#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <time.h>

#define NUM_TESTING_SAMPLE 10000
#define INPUT_NEURONS 784
#define HIDDEN_NEURONS 128
#define OUTPUT_NEURONS 10

using namespace std;

void information();
void init_array();
void load_model(string file_name);
double sigmoid(double x);
void perceptron();
double square_error();

// Testing image file name
const string MNIST_TESTING_IMG = "t10k-images.idx3-ubyte";

// Testing label file name
const string MNIST_TESTING_LABEL = "t10k-labels.idx1-ubyte";

// Weights file name
const string MNIST_MODEL = "model-neural-network.dat";


// Image size in MNIST database
const int width = 28;
const int height = 28;


//  Input layer - Hidden layer
double *w1[INPUT_NEURONS + 1], *out1;

// Hidden layer - Output layer
double *w2[HIDDEN_NEURONS + 1], *iHIDDEN_NEURONS, *out2;

// Output layer
double *iOUTPUT_NEURONS, *out3;
double expected[OUTPUT_NEURONS + 1];

// Image. In MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;


//information printing

void information() {
	// Details
	cout << "*************************************************" << endl;
	cout << "*** Testing Neural Network for MNIST database ***" << endl;
	cout << "*************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << INPUT_NEURONS << endl;
	cout << "No. hidden neurons: " << HIDDEN_NEURONS << endl;
	cout << "No. output neurons: " << OUTPUT_NEURONS << endl;
	cout << endl;
	cout << "Testing image data: " << MNIST_TESTING_IMG << endl;
	cout << "Testing label data: " << MNIST_TESTING_LABEL << endl;
	cout << "No. testing sample: " << NUM_TESTING_SAMPLE << endl << endl;
}

//memory allocation
void init_array() {

	out1 = new double [INPUT_NEURONS + 1];
	iHIDDEN_NEURONS = new double [HIDDEN_NEURONS + 1];
    out2 = new double [HIDDEN_NEURONS + 1];	
    iOUTPUT_NEURONS = new double [OUTPUT_NEURONS + 1];
    out3 = new double [OUTPUT_NEURONS + 1];

    for (int i = 1; i <= INPUT_NEURONS; ++i) {
        w1[i] = new double [HIDDEN_NEURONS + 1];
    }   
    
	
    for (int i = 1; i <= HIDDEN_NEURONS; ++i) {
        w2[i] = new double [OUTPUT_NEURONS + 1];
    }
    
    
}

//load trained model

void load_model(string file_name) {
	ifstream file(file_name.c_str(), ios::in);
	
	// Input layer - Hidden layer
    for (int i = 1; i <= INPUT_NEURONS; ++i) {
        for (int j = 1; j <= HIDDEN_NEURONS; ++j) {
			file >> w1[i][j];
		}
    }
	
	// Hidden layer - Output layer
    for (int i = 1; i <= HIDDEN_NEURONS; ++i) {
        for (int j = 1; j <= OUTPUT_NEURONS; ++j) {
			file >> w2[i][j];
		}
    }
	
	file.close();
}

//sigmoid function

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

//testing
void perceptron() {
    for (int i = 1; i <= HIDDEN_NEURONS; ++i) {
		iHIDDEN_NEURONS[i] = 0.0;
	}

    for (int i = 1; i <= OUTPUT_NEURONS; ++i) {
		iOUTPUT_NEURONS[i] = 0.0;
	}

    for (int i = 1; i <= INPUT_NEURONS; ++i) {
        for (int j = 1; j <= HIDDEN_NEURONS; ++j) {
            iHIDDEN_NEURONS[j] += out1[i] * w1[i][j];
		}
	}

    for (int i = 1; i <= HIDDEN_NEURONS; ++i) {
		out2[i] = sigmoid(iHIDDEN_NEURONS[i]);
	}

    for (int i = 1; i <= HIDDEN_NEURONS; ++i) {
        for (int j = 1; j <= OUTPUT_NEURONS; ++j) {
            iOUTPUT_NEURONS[j] += out2[i] * w2[i][j];
		}
	}

    for (int i = 1; i <= OUTPUT_NEURONS; ++i) {
		out3[i] = sigmoid(iOUTPUT_NEURONS[i]);
	}
}

// +---------------+
// | Norm L2 error |
// +---------------+

double square_error(){
    double res = 0.0;
    for (int i = 1; i <= OUTPUT_NEURONS; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+

int input() {
	// Reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0; 
			} else {
				d[i][j] = 1;
			}
        }
	}

    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
        }
	}

	// Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= OUTPUT_NEURONS; ++i) {
		expected[i] = 0.0;
	}
    expected[number + 1] = 1.0;
        
    return (int)(number);
}

// +--------------+
// | Main Program |
// +--------------+

int main(int argc, char *argv[]) {
	information();
	
    report.open(report_fn.c_str(), ios::out);
    image.open(MNIST_TESTING_IMG.c_str(), ios::in | ios::binary); // Binary image file
    label.open(MNIST_TESTING_LABEL.c_str(), ios::in | ios::binary ); // Binary label file

	// Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
		
	// Neural Network Initialization
    init_array(); // Memory allocation
    load_model(MNIST_MODEL); // Load model (weight matrices) of a trained Neural Network
    
    int nCorrect = 0;
    for (int sample = 1; sample <= NUM_TESTING_SAMPLE; ++sample) {
        cout << "Sample " << sample << endl;
        
        // Getting (image, label)
        int label = input();
		
		// Classification - Perceptron procedure
        perceptron();
        
        // Prediction
        int predict = 1;
        for (int i = 2; i <= OUTPUT_NEURONS; ++i) {
			if (out3[i] > out3[predict]) {
				predict = i;
			}
		}
		--predict;

		// Write down the classification result and the squared error
		double error = square_error();
		printf("Error: %0.6lf\n", error);
		
		if (label == predict) {
			++nCorrect;
			cout << "Classification: YES. Label = " << label << ". Predict = " << predict << endl << endl;
			report << "Sample " << sample << ": YES. Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
		} else {
			cout << "Classification: NO.  Label = " << label << ". Predict = " << predict << endl;
			cout << "Image:" << endl;
			for (int j = 1; j <= height; ++j) {
				for (int i = 1; i <= width; ++i) {
					cout << d[i][j];
				}
				cout << endl;
			}
			cout << endl;
			report << "Sample " << sample << ": NO.  Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
		}
    }

	// Summary
    double accuracy = (double)(nCorrect) / NUM_TESTING_SAMPLE * 100.0;
    cout << "Number of correct samples: " << nCorrect << " / " << NUM_TESTING_SAMPLE << endl;
    printf("Accuracy: %0.2lf\n", accuracy);
    
    report << "Number of correct samples: " << nCorrect << " / " << NUM_TESTING_SAMPLE << endl;
    report << "Accuracy: " << accuracy << endl;

    report.close();
    image.close();
    label.close();
    
    return 0;
}
