#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <time.h>


#define NUM_THREADS 1
#define INPUT_NEURONS 784
#define HIDDEN_NEURONS 128
#define OUTPUT_NEURONS 10
#define NUM_TRAIN_SAMPLES 60000

using namespace std;


int ReverseInt (int i);
void Read_MNIST_training(int NumberOfImages, int DataOfAnImage);
void Read_MNIST_label(int number_of_images,int i);
void information();
void init_global();
double sigmoid(double x);
void save_model(string file_name);


// Training image file name
const string MNIST_TRAIN_IMG = "train-images.idx3-ubyte";

// Training label file name
const string MNIST_TRAIN_LABEL = "train-labels.idx1-ubyte";

// Weights file name
const string MNIST_TRAIN_MODEL = "model-neural-network.dat";


// Image size in MNIST database
int width = 28;
int height = 28;


const int epochs = 1;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *global_w1[INPUT_NEURONS];

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *global_w2[HIDDEN_NEURONS];


double mnist_training_data[NUM_TRAIN_SAMPLES][INPUT_NEURONS];
double mnist_label_data[NUM_TRAIN_SAMPLES][OUTPUT_NEURONS];



//Read MNIST DATASET
int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
void Read_MNIST_training(int NumberOfImages, int DataOfAnImage)
{
    ifstream file (MNIST_TRAIN_IMG,ios::binary);
    if (file.is_open())
    {
      
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        file.read((char*)&magic_number,sizeof(magic_number));

        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    mnist_training_data[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
    file.close();
}

//Read MNIST label
void Read_MNIST_label(int number_of_images,int i)
{
    ifstream file (MNIST_TRAIN_LABEL,ios::binary);

    if (file.is_open())
    {
        
        int num = 0;
        int magic_number = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &num,sizeof(num));
        num = ReverseInt(num);

        for(int img = 0; img < number_of_images; ++img)
        {
            unsigned char temp = 0;

            file.read((char*) &temp, sizeof(temp));

            int number = (double)temp;
            
            for (int i =0; i < OUTPUT_NEURONS; ++i) {
                mnist_label_data[img][i] = 0.0;
            }
            mnist_label_data[img][number] = 1.0;

        }

    }
}

//Print information information neural network
void information() {
	// Details
	cout << "**************************************************" << endl;
	cout << "*** Training Neural Network for MNIST database ***" << endl;
	cout << "**************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << INPUT_NEURONS << endl;
	cout << "No. hidden neurons: " << HIDDEN_NEURONS << endl;
	cout << "No. output neurons: " << OUTPUT_NEURONS << endl;
	cout << endl;
	cout << "No. iterations: " << epochs << endl;
	cout << "Learning rate: " << learning_rate << endl;
	cout << "Momentum: " << momentum << endl;
	cout << "Epsilon: " << epsilon << endl;
	cout << endl;
	cout << "Training image data: " << MNIST_TRAIN_IMG << endl;
	cout << "Training label data: " << MNIST_TRAIN_LABEL << endl;
	cout << "No. training sample: " << NUM_TRAIN_SAMPLES << endl << endl;
    cout << "Num of Threads: " << NUM_THREADS<< endl;
}

//memory allocation

void init_global(){

// Initialization for weights from Input layer to Hidden layer
    for (int i =0; i < INPUT_NEURONS; ++i) {
        global_w1[i] = new double [HIDDEN_NEURONS];
        
        for (int j =0; j < HIDDEN_NEURONS; ++j) {
            
            global_w1[i][j] = 0;
        }
	}
	
	// Initialization for weights from Hidden layer to Output layer
    for (int i =0; i < HIDDEN_NEURONS; ++i) {
        global_w2[i] = new double [OUTPUT_NEURONS];
        
        for (int j =0; j < OUTPUT_NEURONS; ++j) {
            
            global_w2[i][j] = 0;
        }
	}
}


//sigmoid activaiton function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


//save training model
void save_model(string file_name) {

    ofstream file(file_name.c_str(), ios::out);
	
	// Input layer - Hidden layer
    for (int i =0; i < INPUT_NEURONS; ++i) {
        for (int j =0; j < HIDDEN_NEURONS; ++j) {
			file << global_w1[i][j] << " ";
		}
		file << endl;
    }
	
	// Hidden layer - Output layer
    for (int i =0; i < HIDDEN_NEURONS; ++i) {
        for (int j =0; j < OUTPUT_NEURONS; ++j) {
			file << global_w2[i][j] << " ";
		}
        file << endl;
    }
	
	file.close();
}

//main function
int main(int argc, char *argv[]) {

    //read MNIST DATASET
    Read_MNIST_training(60000,INPUT_NEURONS);
    Read_MNIST_label(60000,10);
    
    //print information
    information();
		
	//Begin time
    time_t begin = time(NULL);
    cout <<"Begin time = " << begin <<"\n";

    // Neural Network Initialization
	init_global();


    int sample_per_thread = NUM_TRAIN_SAMPLES / NUM_THREADS;

    #pragma omp parallel num_threads(NUM_THREADS) default(none) shared(cout,mnist_label_data,mnist_training_data,sample_per_thread,global_w2,global_w1)
    {
        // From  Input layer - Hidden layer
        double w1[INPUT_NEURONS][HIDDEN_NEURONS], delta1[INPUT_NEURONS][HIDDEN_NEURONS], out1[INPUT_NEURONS];

        // From  Hidden layer - Output layer
        double w2[HIDDEN_NEURONS][OUTPUT_NEURONS], delta2[HIDDEN_NEURONS][OUTPUT_NEURONS], iHIDDEN_NEURONS[HIDDEN_NEURONS], out2[HIDDEN_NEURONS], theta2[HIDDEN_NEURONS];

        // Output layer
        double iOUTPUT_NEURONS[OUTPUT_NEURONS], out3[OUTPUT_NEURONS], theta3[OUTPUT_NEURONS];
        //target label
        double expected[OUTPUT_NEURONS];


        for (int i =0; i < INPUT_NEURONS; ++i) {
            for (int j =0; j < HIDDEN_NEURONS; ++j) {

                w1[i][j] = (double)(rand() % 6) / 10.0;
                
            }
        }
        // Initialization for weights from Hidden layer to Output layer
        for (int i =0; i < HIDDEN_NEURONS; ++i) {
            for (int j =0; j < OUTPUT_NEURONS; ++j) {
                
                w2[i][j] = (double)(rand() % 10) / 10.0 ;
                
            }
        }


        #pragma omp for 

        for (int sample =0; sample < 60000; ++sample) {
            
            // input to neural network
            for (int i =0; i < INPUT_NEURONS; ++i) {
                out1[i] = mnist_training_data[sample][i];
            }
            // target output
            for (int i = 0; i < OUTPUT_NEURONS; ++i) {
                expected[i] = mnist_label_data[sample][i];
            }


    		// Learning process
            for (int i =0; i < INPUT_NEURONS; ++i) {
                for (int j =0; j < HIDDEN_NEURONS; ++j) {
                    delta1[i][j] = 0.0;
                }
            }

            for (int i =0; i < HIDDEN_NEURONS; ++i) {
                for (int j =0; j < OUTPUT_NEURONS; ++j) {
                    delta2[i][j] = 0.0;
                }
            }

            for (int i =0; i < epochs; ++i) {
                // forward 
                for (int i =0; i < HIDDEN_NEURONS; ++i) {
                    iHIDDEN_NEURONS[i] = 0.0;
                }

                for (int i =0; i < OUTPUT_NEURONS; ++i) {
                    iOUTPUT_NEURONS[i] = 0.0;
                }

                for (int i =0; i < INPUT_NEURONS; ++i) {
                    for (int j =0; j < HIDDEN_NEURONS; ++j) {
                        iHIDDEN_NEURONS[j] += out1[i] * w1[i][j];
                    }
                }

                for (int i =0; i < HIDDEN_NEURONS; ++i) {
                    out2[i] = sigmoid(iHIDDEN_NEURONS[i]);
                }

                for (int i =0; i < HIDDEN_NEURONS; ++i) {
                    for (int j =0; j < OUTPUT_NEURONS; ++j) {
                        iOUTPUT_NEURONS[j] += out2[i] * w2[i][j];
                    }
                }

                for (int i =0; i < OUTPUT_NEURONS; ++i) {
                    out3[i] = sigmoid(iOUTPUT_NEURONS[i]);
                }

                // back prop
                double sum;
                for (int i =0; i < OUTPUT_NEURONS; ++i) {
                    theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
                }

                for (int i =0; i < HIDDEN_NEURONS; ++i) {
                    sum = 0.0;
                    for (int j =0; j < OUTPUT_NEURONS; ++j) {
                        sum += w2[i][j] * theta3[j];
                    }
                    theta2[i] = out2[i] * (1 - out2[i]) * sum;
                }

                for (int i =0; i < HIDDEN_NEURONS; ++i) {
                    for (int j =0; j < OUTPUT_NEURONS; ++j) {
                        delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
                        w2[i][j] += delta2[i][j];
                        
                    }
                }

                for (int i =0; i < INPUT_NEURONS; ++i) {
                    for (int j = 0 ; j < HIDDEN_NEURONS ; j++ ) {
                        delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
                        w1[i][j] += delta1[i][j];
                        
                    }
                }
            }
            if((sample + 1) % sample_per_thread == 0)
            {
                for (int i =0; i < INPUT_NEURONS; ++i) {
                    for (int j = 0; j < HIDDEN_NEURONS ; j++ ) {
                        #pragma omp critical
                        {
                            global_w1[i][j] += w1[i][j];
                        }
                    }
                }
                for (int i =0; i < HIDDEN_NEURONS; ++i) {
                    for (int j = 0; j < OUTPUT_NEURONS ; j++ ) {
                        #pragma omp critical
                        {
                            global_w2[i][j] += w2[i][j];
                        }
                    }
                }
            }
            
        }
    }

    //end time
	time_t end = time(NULL);
    double elapsed_secs = difftime(end,begin);

    //average thread weights
    for (int i =0; i < INPUT_NEURONS; ++i) {
        for (int j = 0; j < HIDDEN_NEURONS ; j++ ) {
           global_w1[i][j] /= NUM_THREADS;
            
        }
    }
    for (int i =0; i < HIDDEN_NEURONS; ++i) {
        for (int j = 0; j < OUTPUT_NEURONS ; j++ ) {
          global_w2[i][j] /= NUM_THREADS;
            
        }
    }

	// Save the final network
    save_model(MNIST_TRAIN_MODEL);

    //print elapsed time
    cout << "Elapsed time: " << elapsed_secs  << endl;
    
    return 0;
}

