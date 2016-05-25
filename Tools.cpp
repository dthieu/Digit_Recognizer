#include "Tools.h"
#include <fstream>

bool CTools::ReadDataTrain(string fileNameIn, vector<Mat> &dataTrain, vector<int>& labels){

	fstream file;

	string header;

	dataTrain.clear();
	labels.clear();

	//open file
	file.open(fileNameIn, ios::in);
	if (!file.is_open())
		return false;
	//read header of file;
	getline(file, header);

	char limit;
	int label;
	int pixel;

	 //read labels
	while (file >> label)
	{
		Mat image(N_ROWS,N_COLS,CV_8U);

		
		
		for ( int m = 0; m < N_ROWS; m++)
		{
			for ( int n = 0; n < N_COLS; n++)
			{
				file >> limit >>  pixel;//read limit (",") and pixel ( 0-> 255)
				image.at<uchar>(m,n) = (uchar)pixel;
			}
		}

		dataTrain.push_back(image);
		labels.push_back(label);
	}
	file.close();
	return true;
}


bool CTools::ReadDataTest(string fileNameIn, vector<Mat> &dataTest){

	fstream file;
	string header;

	//open file
	file.open(fileNameIn, ios::in);
	if (!file.is_open())
		return false;
	//read header of file;
	getline(file, header);

	char limit;
	int pixel;

	while ( !file.eof()) 
	{
		Mat A(N_ROWS, N_COLS,CV_8U);
		for (int m = 0; m < N_ROWS; m++){
			for ( int n = 0; n < N_COLS; n++){
				if (file >> pixel)
				{
					A.at<uchar>(m,n) = (uchar)pixel;

					if ( m * N_COLS + n != 783 )
						file >> limit;
				}else
					return true;
			}
		}
		dataTest.push_back(A);
	}
	file.close();
	return true;
}

void CTools::WriteResult(vector<int> result, string fileNameOut){
	fstream file;
	string header = "\"ImageId\",\"Label\"\n";
	file.open(fileNameOut, ios::out);
	
	file << header;
	for (unsigned int ImgId = 0; ImgId < result.size(); ImgId++){
		file << ImgId + 1 << ",\"" <<  result[ImgId] << "\"" <<'\n';
	}
	
	file.close();
}





 int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


bool CTools::read_Mnist(string filename, vector<Mat> &vec){
    ifstream file (filename, ios::binary);

    if (!file.is_open())
		return false;
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    file.read((char*) &magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);

    file.read((char*) &number_of_images,sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);

    file.read((char*) &n_rows, sizeof(n_rows));
    n_rows = ReverseInt(n_rows);

    file.read((char*) &n_cols, sizeof(n_cols));
    n_cols = ReverseInt(n_cols);

    for(int i = 0; i < number_of_images; ++i)
    {
		cv::Mat tp = cvCreateMat(n_rows, n_cols, CV_8U);
        for(int r = 0; r < n_rows; ++r)
        {
            for(int c = 0; c < n_cols; ++c)
            {
                unsigned char temp = 0;
                file.read((char*) &temp, sizeof(temp));
                tp.at<uchar>(r, c) = (uchar) temp;
            }
        }
        vec.push_back(tp);
    }

	file.close();
	return true;
}

bool CTools::read_Mnist_Label(string filename, vector<int> &vec)
{
    ifstream file (filename, ios::binary);
    if (!file.is_open())
		return false;

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    file.read((char*) &magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);

    file.read((char*) &number_of_images,sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);

    for(int i = 0; i < number_of_images; ++i)
    {
        unsigned char temp = 0;
        file.read((char*) &temp, sizeof(temp));
		vec.push_back((int)temp);
    }
    
	file.close();
	return true;
}




void CTools::ConvertListFeatureToMat(vector<vector<float>> &features, Mat& featuresMat){
	featuresMat.release();
	unsigned int numF = features.size();
	unsigned int dimF = features[0].size();
	featuresMat = cvCreateMat((int)numF, (int)dimF, CV_32FC1);
	for ( unsigned int i = 0; i < numF; i++){
		for ( unsigned int k = 0; k < dimF; k++) {
			featuresMat.at<float>(i, k) = features[i][k];
		}
	}
}


Mat CTools::CombinedImage ( vector<Mat> listImage,unsigned int number){
	
	cv::Size sb_size = listImage[0].size();
	int t = number / 10;

	cv::Mat combined(t * sb_size.height, 10 * sb_size.width, listImage[0].type());

	unsigned int i = 0;
		for ( int x = 0; x < combined.cols; x += sb_size.height)
		{
			for ( int y = 0; y < combined.rows; y += sb_size.width, i++)
			{
				cv::Mat roi = combined(cv::Rect(x,y,sb_size.height,sb_size.width));
				listImage[i].copyTo(roi);
			}
		}

	return combined;
}

void CTools::Merge(vector<vector<float>> &features1, vector<vector<float>> &features2, Mat& featuresMat){
	featuresMat.release();
	unsigned int numF1 = features1.size();
	unsigned int numF2 = features2.size();
	unsigned int numF = numF1 + numF2;
	if (features1[0].size() != features2[0].size())
		return;

	unsigned int dimF = features1[0].size();

	featuresMat = cvCreateMat((int)numF, (int)dimF, CV_32FC1);

	for ( unsigned int i = 0; i < numF1; i++){
		for ( unsigned int k = 0; k < dimF; k++) {
			featuresMat.at<float>(i, k) = features1[i][k];
		}
	}

	for ( unsigned int i = 0; i < numF2; i++){
		for ( unsigned int k = 0; k < dimF; k++) {
			featuresMat.at<float>(i + numF1, k) = features2[i][k];
		}
	}

}