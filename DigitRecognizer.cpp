#pragma once
#include "Testing.h"
#include "Training.h"
#include <opencv\highgui.h>
#include <time.h>
CvSVMParams CreateParamSVM(int);


//for SVM param
static int Iter = 1000;

Rate rate;
/*Usage: DigitRecognizer.exe models rate function fileNameAction fileNameAfter [fileResult]
Models [0/1]: bayes/SVM 
rate [0/1/2]: LOW/ MEDIUM/ HIGHT
function: [0/1] train/Test

if function is 0 (Train) then:
	fileNameAction: fileName of data Train (*.csv)
	fileNameAfter: fileName save dataTrain (*.xml)

else function is 1 (Test):

	filenameAction: fileName of data Test (*.csv)
	fileNameAfter: fileName save dataTrain (*.xml)
	fileResult: fileName result of program. (*.csv)
*/


int main(int argc, char *argv[]){
	
	if ( argc != 6 && argc != 7 )
	{
		cout << "Please read the ReadMe file" << endl;
		return 1;
	}

	int imodels = atoi ( argv[1]);
	int irate = atoi ( argv[2]);
	int ifunction = atoi ( argv[3]);
	string data(argv[4]);
	string saveDataTrain(argv[5]);
	Models models;
	Rate rate;
	//default argument
	if (imodels == 1 )
	{
		models =  Models::_SVM;
	}
	else
	{
		models = Models::_BAYES;
	}

	if (irate == 1 )
	{
		rate = Rate::MEDIUM;
	}else if (irate == 2 )
	{
		rate = Rate::HIGH;
	}else
	{
		rate = Rate::LOW;
	}


	

	switch (ifunction)
	{
		//test
	case 1:
		{
			string resultFile(argv[6]);

			vector<Mat> dataTest;
			cout << "Reading data..." << endl;
			if (CTools::ReadDataTest(data, dataTest))
			{



				CFeature act(rate);
				vector<vector<float>> features;
				cout << "Extract features..." << endl;
				act.HOGfeatureDetector(dataTest, features);


				Mat allFeatures;
				CTools::ConvertListFeatureToMat(features, allFeatures);

				vector<int> results;
				CTesting testing(models);
				cout << "Generate output..." << endl;

				clock_t start = clock();
				testing.testListImage(saveDataTrain,  allFeatures, results);

				cout << "Writing a result to output file..." << endl;
				CTools::WriteResult(results, resultFile);

				clock_t end = clock();
				cout << "Time Testing: " << (double) (end - start) / CLOCKS_PER_SEC << end;


				

			}
			else
			{
				cout << "File or directory is corrupted and unreadable";
			}
			break;
		}
		//train
	default:
		{
			vector<Mat> dataTrain;
			vector<int> vecLabels;
			cout << "Reading data..." << endl;
			if (CTools::ReadDataTrain(data, dataTrain, vecLabels))
			{

				//Show 100 image and labels
				/*
				cout << endl;
				for ( int i = 0; i < 10; i++)
				{
					for ( int j = 0; j < 10; j++)
					{
						cout << vecLabels[j * 10 + i] << " ";
					}
					cout << endl;
				}

				Mat cb = CTools::CombinedImage(dataTrain, 100);
				imshow("cb", cb);
				waitKey(0);
				cvDestroyWindow("cb");
				*/

				CFeature act(rate);
				vector<vector<float>> features;
				cout << "Extract features..." << endl;
				act.HOGfeatureDetector(dataTrain, features);
				cout << "Dimension feature: " << features[0].size() << endl;
				Mat allFeatures;
				Mat allLabels(vecLabels);
				CTools::ConvertListFeatureToMat(features, allFeatures);

				CTrainingData TrainData(saveDataTrain);
				
				cout << "Trainging..." << endl;
				clock_t start = clock();
				switch (models)
				{
				case _BAYES:
					TrainData.TrainingForBayes(allFeatures, allLabels);
					break;
				case _SVM:
					TrainData.TrainingForSVM(allFeatures, allLabels, CreateParamSVM(Iter));
					break;
				case _EMPTY:
					break;
				default:
					break;
				}
				clock_t end = clock();
				cout << "Time Training: " << (double) (end - start) / CLOCKS_PER_SEC << end;
			}
			else
			{
				cout << "File or directory is corrupted and unreadable";
			}
			break;
		}
	}

	return 0;

}

//iter:: số lần lặp của SVM
//Hoặc theo EPSILON là  FLT_EPSILON
CvSVMParams CreateParamSVM(int iter){
	
	CvTermCriteria term = cvTermCriteria(CV_TERMCRIT_ITER +CV_TERMCRIT_EPS, iter, FLT_EPSILON);
	CvSVMParams param = CvSVMParams();
	param.svm_type = CvSVM::C_SVC;
	param.kernel_type = CvSVM::LINEAR;
	param.term_crit = term;
	return param;
}
