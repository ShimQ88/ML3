#ifndef LOAD_AND_SAVE_ML_H
#define LOAD_AND_SAVE_ML_H

// System Headers
#include <iostream>
#include <cstdio>
#include <vector>
#include <fstream>
#include <string>
#include "opencv2/ml.hpp"
// Opencv Headers
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"



//namespace
using namespace std;
using namespace cv;
using namespace cv::ml;

bool
load_and_save_ml( const string& data_filename,
                      const string& filename_to_save,
                      const string& filename_to_load,
                      float percent_of_division,
                      int ml_technique);
bool
read_num_class_data( const string& filename, int var_count,
                     Mat* _data, Mat* _responses );

void 
help(bool swap_the_role_train_to_test,int ntrain_samples, int ntest_samples, int ml_technique);

int
Count_Column_Numb(const string& filename);

class Machine_Learning_Data_Preparation{
// protected:
    // int hello;
public:
    Mat data;
    Mat responses;
    int ntrain_samples;
    int ntest_samples;
    int class_count;
    Ptr<TrainData> *tdata;
    string filename_to_save;
    string filename_to_load;

    int k_fold_value;
    int n_total_samples;
    int the_number_of_data;
    int block;
    int value;
    int ml_technique;
    // Ptr<TrainData> tdata;

    // int t(){
    //  return hello;
    // }
    Mat *train_data;
    Mat *test_data;
    Mat *train_responses;
    Mat *test_responses;
    Mat *train_responses_int;
    Mat *test_responses_int;
// public:
    Machine_Learning_Data_Preparation(){cout<<"Success Intialize"<<endl;}
    ~Machine_Learning_Data_Preparation(){
        delete tdata[k_fold_value];
        delete train_data;
        delete test_data;
        delete train_responses;
        delete test_responses;
        delete train_responses_int;
        delete test_responses_int;
    }

    Machine_Learning_Data_Preparation(Mat i_data, Mat i_responses, 
        int i_ntrain_samples, int i_ntest_samples, 
        const string& i_filename_to_save, const string& i_filename_to_load, int the_number_of_class);

    // void Split(int input_value,int technique){
    void Main_Process(int technique);
    bool Split_train_test_data(Mat *train_data, Mat *test_data, Mat *train_responses, Mat *test_responses,
                        Mat *train_responses_int, Mat *test_responses_int, Mat data, Mat responses,
                        int block,int the_number_of_data, int n_total_samples, int ntest_samples, int technique);

    inline TermCriteria TC(int iters, double eps);
};

Machine_Learning_Data_Preparation::Machine_Learning_Data_Preparation(Mat i_data, Mat i_responses, 
    int i_ntrain_samples, int i_ntest_samples, 
    const string& i_filename_to_save, const string& i_filename_to_load, int the_number_of_class){    
    // cout<<"filename_to_save: "<<&filename_to_save<<endl;
    data=i_data;
    responses=i_responses;
    ntrain_samples=i_ntrain_samples;
    ntest_samples=i_ntest_samples;
    filename_to_save=i_filename_to_save;
    filename_to_load=i_filename_to_load;

    class_count=the_number_of_class;

    the_number_of_data=data.cols;
    // cout<<"the_number_of_data: "<<the_number_of_data<<endl;
    k_fold_value=ntrain_samples/ntest_samples;
    n_total_samples=ntrain_samples+ntest_samples;
    // train_data = Mat::zeros( p_ntrain_samples, the_number_of_data, CV_32F );
    //      test_data = Mat::zeros( p_ntest_samples, the_number_of_data, CV_32F );


    //      train_responses = Mat::zeros( p_ntrain_samples, p_class_count, CV_32F );
    // test_responses = Mat::zeros( p_ntest_samples, p_class_count, CV_32F );
    //       train_responses_int = Mat::zeros( p_ntrain_samples, 1, CV_32F );
    // test_responses_int = Mat::zeros( p_ntest_samples, 1, CV_32F );
}
void Machine_Learning_Data_Preparation::Main_Process(int technique){
    //Split data
    value=0;
    cout<<"train_responses: "<<train_responses<<endl;
    // cout<<"p_class_count: "<<p_class_count<<endl;
    k_fold_value=ntrain_samples/ntest_samples;
    n_total_samples=ntrain_samples+ntest_samples;
    ml_technique=technique;
    train_data=new Mat[k_fold_value];
    test_data=new Mat[k_fold_value];
    train_responses=new Mat[k_fold_value];
    test_responses=new Mat[k_fold_value];
    train_responses_int=new Mat[k_fold_value];
    test_responses_int=new Mat[k_fold_value];
    
    tdata=new Ptr<TrainData>[k_fold_value];

    while(1){
        if(value==k_fold_value){break;}
        train_data[value] = Mat::zeros( ntrain_samples, the_number_of_data, CV_32F );
        test_data[value] = Mat::zeros( ntest_samples, the_number_of_data, CV_32F );
        if(ml_technique==0){
            train_responses[value] = Mat::zeros( ntrain_samples, class_count, CV_32F );
            test_responses[value] = Mat::zeros( ntest_samples, class_count, CV_32F );
            train_responses_int[value] = Mat::zeros( ntrain_samples, 1, CV_32F );
            test_responses_int[value] = Mat::zeros( ntest_samples, 1, CV_32F );
        }else if(ml_technique==1){
            // train_responses[value] = Mat::zeros( ntrain_samples, class_count, CV_32S );
            // test_responses[value] = Mat::zeros( ntest_samples, class_count, CV_32S );
      //       train_responses_int[value] = Mat::zeros( ntrain_samples, 1, CV_32S );
            // test_responses_int[value] = Mat::zeros( ntest_samples, 1, CV_32S );

            train_responses[value] = Mat::zeros( ntrain_samples, class_count, CV_32S);
            test_responses[value] = Mat::zeros( ntest_samples, class_count, CV_32S );
            train_responses_int[value] = Mat::zeros( ntrain_samples, 1, CV_32S );
            test_responses_int[value] = Mat::zeros( ntest_samples, 1, CV_32S );
        }else if(ml_technique==2){
            train_responses[value] = Mat::zeros( ntrain_samples, class_count, CV_32F );
            test_responses[value] = Mat::zeros( ntest_samples, class_count, CV_32F );
            train_responses_int[value] = Mat::zeros( ntrain_samples, 1, CV_32F );
            test_responses_int[value] = Mat::zeros( ntest_samples, 1, CV_32F );

            // train_responses[value] = Mat::zeros( ntrain_samples, class_count, CV_32FC1 );
            // test_responses[value] = Mat::zeros( ntest_samples, class_count, CV_32FC1 );
            // train_responses_int[value] = Mat::zeros( ntrain_samples, 1, CV_32FC1 );
            // test_responses_int[value] = Mat::zeros( ntest_samples, 1, CV_32FC1 );
            
        }else{
            
        }
        
        // cout<<"p_ntest_samples: "<<p_ntest_samples<<endl;
        int block=ntest_samples*value;


        Split_train_test_data(&train_data[value], &test_data[value], &train_responses[value], &test_responses[value],
                    &train_responses_int[value], &test_responses_int[value], data, responses,
                    block, the_number_of_data, n_total_samples, ntest_samples,ml_technique);

        
        if(ml_technique==0){//ANN
            tdata[value] = TrainData::create(train_data[value], ROW_SAMPLE, train_responses[value]);//train_responses: 2col many rows
        }else if(ml_technique==1){//Boost
            tdata[value] = TrainData::create(train_data[value], ROW_SAMPLE, train_responses_int[value]);//train_responses_int: 1 col many rows
            // tdata[value] = TrainData::create(train_data[value], ROW_SAMPLE, train_responses[value]);//train_responses_int: 1 col many rows
        }else if(ml_technique==2){//RF
            tdata[value] = TrainData::create(train_data[value], ROW_SAMPLE, train_responses_int[value]);
        }else{

        }
        value++;
        // cout<<"train_data: "<<train_data[]<<endl;
        // getchar();
    }
    // cout<<"k_fold_value: "<<k_fold_value<<endl;
    // getchar();
    // tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);
    // tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses_int);
        // cout<<"block: "<<block<<endl;

        // cout<<"test_responses: "<<test_responses<<endl;
        // cout<<"end"<<endl;
        // getchar();
    }
    bool Machine_Learning_Data_Preparation::Split_train_test_data(Mat *train_data, Mat *test_data, Mat *train_responses, Mat *test_responses,
                    Mat *train_responses_int, Mat *test_responses_int, Mat data, Mat responses,
                    int block,int the_number_of_data, int n_total_samples, int ntest_samples, int technique)
{
    int i_train=0;
    int i_test=0;

    

   for(int i=0;i<n_total_samples;i++){
        int cls_label = responses.at<int>(i) - 48;// - 'A'; //change to numerical classes, still they read as chars
        cout << "labels " << cls_label << endl;
        if( (i>=block)&&(i<block+ntest_samples) ){
            if(ml_technique==1){
                if(cls_label==1){
                    test_responses->at<int>(i_test, cls_label) = 1;
                    // test_responses_int->at<int>(i_test,0)=cls_label;
                    test_responses_int->at<int>(i_test,0)=1;
                }
            }else{
                test_responses->at<float>(i_test, cls_label) = 1.f;
                test_responses_int->at<float>(i_test,0)=cls_label;
            }
            
        }else{//test part
            if(ml_technique==1){
                if(cls_label==1){
                    train_responses->at<int>(i_train, cls_label) = 1;
                    // train_responses_int->at<int>(i_train,0)=cls_label; 
                    train_responses_int->at<int>(i_train,0)=1; 
                }
            }else{
                train_responses->at<float>(i_train, cls_label) = 1.f;
                train_responses_int->at<float>(i_train,0)=cls_label; 
            }
            
        }
        for(int j=0;j<the_number_of_data;j++){
            if( (i>=block)&&(i<block+ntest_samples) ){
                if(ml_technique==1){
                    test_data->at<int>(i_test,j)=data.at<int>(i,j);
                }else{
                    test_data->at<float>(i_test,j)=data.at<float>(i,j); 
                }
            }else{
                if(ml_technique==1){
                    train_data->at<int>(i_train,j)=data.at<int>(i,j);
                }else{
                    train_data->at<float>(i_train,j)=data.at<float>(i,j);   
                }
                
            }
        }
        if( (i>=block)&&(i<block+ntest_samples) ){
            i_test++;
        }else{
            i_train++;
                
        }
    }
 //    cout<<"n_total_samples: "<<n_total_samples<<endl;
 //    cout<<"end"<<endl;
 //    // getchar();

 //    cout<<"i_train: "<<i_train<<endl;
    // cout<<"i_test: "<<i_test<<endl;
    // getchar();
    return true;
}
inline TermCriteria Machine_Learning_Data_Preparation::TC(int iters, double eps){
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

template <class T>
T *Creat_Default_ML_Class(){
    return new T();
}

template <class T>
T *Creat_ML_Class(int p1, int p2, float p3, int p4, int p5){
    return new T(p1, p2, p3, p4, p5);
}
// template <class T>
// T *Creat_ML_Class(double p1, int p2, double p3){
//     return new T(p1, p2, p3);
// }
// template <class T>
// T *Creat_ML_Class(double p1, int p2, double p3){
//     return new T(p1, p2, p3);
// }
// template <class T>
// T *Creat_ML_Class(double p1, int p2, double p3){
//     return new T(p1, p2, p3);
// }





class Parent_ML{
public:
    
    Mat *confusion_matrix;
    Mat *confusion_matrix_test;
    Mat *confusion_matrix_train;

    float *accuracy;
    float *accuracy_test;
    float *accuracy_train;
    // char **result_buffer;
    float sum_accuracy=0;
    float sum_accuracy_test=0;
    float sum_accuracy_train=0;

    int ml_technique;
    float mean;
    float mean_test;
    float mean_train;
    
    float variance=0;
    float variance_test=0;
    float variance_train=0;
    
    float sta_dev;
    float sta_dev_test;
    float sta_dev_train;

    char **result_buffer;
    char **result_buffer_test;
    char **result_buffer_train;

    char *final_result_buffer;
    char *final_result_buffer_test;
    char *final_result_buffer_train;

    Machine_Learning_Data_Preparation *ml;
    Parent_ML();
    ~Parent_ML();
    void Main_Process(Machine_Learning_Data_Preparation *&prepared_data);
    float Accuracy_Calculation(const Mat& confusion_matrix);
    bool Calculate_standard_deviation();
    float Calculate_Mean(float i_sum_accuracy, Machine_Learning_Data_Preparation *i_ml);
    float Calculate_Variance(float i_accuracy[],float i_mean, Machine_Learning_Data_Preparation *i_ml);
    float Calculate_Standard_Deviation(float i_variance);
    
    Mat test_and_save_classifier(const Ptr<StatModel>& model,const Mat& data, const Mat& responses, int ntrain_samples, int rdelta, const string& filename_to_save, int ml_technique);
    virtual void Intialize()=0;
    virtual Mat *Calculate_Confusion_Matrices(int type, Machine_Learning_Data_Preparation *&i_ml)=0;
    virtual float *Calculate_Accuracies(Mat *&i_confusion_matrix, Machine_Learning_Data_Preparation *&i_ml)=0;
    virtual float Calculate_Sum_Accuracy(float *&i_accuracy, Machine_Learning_Data_Preparation *&i_ml)=0;
    virtual void Calculate_Result()=0;
    virtual void Return_Parameter(int index)=0;
    virtual string Head_Parameter()=0;
};

Parent_ML::Parent_ML(){
}

void Parent_ML::Main_Process(Machine_Learning_Data_Preparation *&prepared_data){
    ml=prepared_data;

    confusion_matrix=new Mat[ml->k_fold_value];
    confusion_matrix_test=new Mat[ml->k_fold_value];
    confusion_matrix_train=new Mat[ml->k_fold_value];

    accuracy=new float[ml->k_fold_value];
    accuracy_test=new float[ml->k_fold_value];
    accuracy_train=new float[ml->k_fold_value];

    result_buffer=new char*[ml->k_fold_value];
    result_buffer_test=new char*[ml->k_fold_value];
    result_buffer_train=new char*[ml->k_fold_value];

    final_result_buffer=new char[50];
    final_result_buffer_test=new char[50];
    final_result_buffer_train=new char[50];
    

    for(int i=0;i<ml->k_fold_value;i++){
        result_buffer[i]=new char[50];
    }
    for(int i=0;i<ml->k_fold_value;i++){
        result_buffer_test[i]=new char[50];
    }
    for(int i=0;i<ml->k_fold_value;i++){
        result_buffer_train[i]=new char[50];
    }

    // cout<<"done read structure"<<endl;
    // getchar();
    Intialize();
    // Calculate_Result();
    // Calculate_standard_deviation();

    //virtual functions
    confusion_matrix_test=Calculate_Confusion_Matrices(0,ml);
    accuracy_test=Calculate_Accuracies(confusion_matrix_test, ml);
    sum_accuracy_test=Calculate_Sum_Accuracy(accuracy_test, ml);

    // cout<<"confusion_matrix_test: "<<confusion_matrix_test[0]<<endl;
    // getchar();
    // confusion_matrix_train=Calculate_Confusion_Matrices(ml);
    // accuracy_train=Calculate_Accuracies(confusion_matrix_train, ml);
    // sum_accuracy_train=Calculate_Sum_Accuracy(accuracy_train, ml);
    
    //parent functions
    mean_test=Calculate_Mean(sum_accuracy_test, ml);
    variance_test=Calculate_Variance(accuracy_test, mean_test, ml);
    sta_dev_test=Calculate_Standard_Deviation(variance_test);

    // mean_train=Calculate_Mean(sum_accuracy_train, ml);
    // variance_train=Calculate_Variance(accuracy_train, mean_train, ml);
    // sta_dev_train=Calculate_Standard_Deviation(variance_train);


}

Parent_ML::~Parent_ML(){
    for(int i=0;i<ml->k_fold_value;i++){
        delete result_buffer[i];
    }
    for(int i=0;i<ml->k_fold_value;i++){
        delete result_buffer_test[i];
    }
    for(int i=0;i<ml->k_fold_value;i++){
        delete result_buffer_train[i];
    }

    delete accuracy;
    delete accuracy_test;
    delete accuracy_train;

    delete confusion_matrix;
    delete confusion_matrix_test;
    delete confusion_matrix_train;
}

float Parent_ML::Calculate_Mean(float i_sum_accuracy, Machine_Learning_Data_Preparation *i_ml){
    float out_mean=i_sum_accuracy/i_ml->k_fold_value;
    return out_mean;
}

float Parent_ML::Calculate_Variance(float i_accuracy[],float i_mean, Machine_Learning_Data_Preparation *i_ml){
    float out_variance=0;
    for(int j=0; j < i_ml->k_fold_value; j++){
        out_variance=out_variance+(i_accuracy[j]-i_mean)*(i_accuracy[j]-i_mean);
    }
    out_variance=out_variance/i_ml->k_fold_value;
    return out_variance;
}

float Parent_ML::Calculate_Standard_Deviation(float i_variance){
    float out_sta_dev=sqrt(i_variance);
    return out_sta_dev;
}

float Parent_ML::Accuracy_Calculation(const Mat& i_confusion_matrix){
    // load classifier from the specified file
    float out_accuracy;
    float total_accurate=i_confusion_matrix.at<int>(0,0)+i_confusion_matrix.at<int>(1,1);
    float total_number_of_values=i_confusion_matrix.at<int>(0,0)+i_confusion_matrix.at<int>(0,1)+
    i_confusion_matrix.at<int>(1,0)+i_confusion_matrix.at<int>(1,1);
    out_accuracy=total_accurate/total_number_of_values;
    return out_accuracy;
}

Mat Parent_ML::test_and_save_classifier(const Ptr<StatModel>& model,
                                     const Mat& data, const Mat& responses,
                                     int ntrain_samples, int rdelta,
                                     const string& filename_to_save, int ml_technique )
{
    // int nsamples_all = data.rows;
    int nsamples_all = ntrain_samples;
    double train_hr = 0, test_hr = 0;
    int training_correct_predict=0;
    // compute prediction error on training data
    // for(int i=0; i<nsamples_all; i++){
    Mat confusion_Matrix = Mat::zeros( 2, 2, CV_32S );
    
    // getchar();
    int problem_val=0;
    for(int i=0; i<nsamples_all; i++){
        Mat sample = data.row(i);
        // int actual_value=responses.at<int>(i)-48;
        int actual_value;
        if(ml_technique==1){
            actual_value=responses.at<int>(i);
        }else{
            actual_value=responses.at<float>(i);
        }
        // int actual_value=responses.at<int>(i);//this is work for tech 1
        // cout<<"actual_value: "<<responses.at<int>(i)<<endl;
        // getchar();
        // cout << "Actual_ha: " << responses.at<float>(i) << " row " << sample << endl;
        cout << "Actual: " << actual_value << endl;
        // cout<<"sample: "<<sample<<endl;
        float r = model->predict( sample );
        cout<<"predict: "<<r<<endl;
        // getchar();
        // cout << "Predict:  r = " << round(r) << endl;//rounding in case of random_forest
        // getchar();
        // int r_int=r;
        int r_int=(int)round(r);//random forrest case
        // if( r_int == actual_value ){ //prediction is correct
        if( r_int == actual_value ){ //prediction is correct
            training_correct_predict++;
        }
        confusion_Matrix.at<int>(actual_value,r_int)=confusion_Matrix.at<int>(actual_value,r_int)+1;
        // if(actual_value==0){
        //     if(r>0.3){
        //         cout<<"This "<<r<<" value is problem"<<endl;
        //         problem_val++;
        //     }else{

        //     }
        // }else{
        //     if(r<0.7){
        //         cout<<"This "<<r<<" value is problem"<<endl;
        //         problem_val++;
        //     }else{
                
        //     }
        // }
        


        // cout<<"confusion_Matrix: "<<confusion_Matrix<<endl;
        // getchar();


        // cout << "training_correct_predict = " << training_correct_predict << endl;
        // getchar();

        // cout << "Sample: " << responses.at<int>(i) << " row " << data.row(i) << endl;
        // float r = model->predict( sample );
        // cout << "Predict:  r = " << r << endl;
        // if( (int)r == (int)(responses.at<int>(i)) ) //prediction is correct
        //     training_correct_predict++;
   
    // r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
    
     
        //if( i < ntrain_samples )
        //    train_hr += r;
        //else
        //    test_hr += r;

    }

    //test_hr /= nsamples_all - ntrain_samples;
    //train_hr = ntrain_samples > 0 ? train_hr/ntrain_samples : 1.;
    printf("ntrain_samples %d training_correct_predict %d \n",ntrain_samples, training_correct_predict);
    // *accuracy=training_correct_predict*100.0/ntrain_samples;
    // getchar();
    if( filename_to_save.empty() )  {
        printf( "\nTest Recognition rate: training set = %.1f%% \n\n", training_correct_predict*100.0/ntrain_samples);

    }
    // if( filename_to_save.empty() )  printf( "\nTest Recognition rate: training set = %.1f%% \n\n", *accuracy);


    if( !filename_to_save.empty() )
    {
        model->save( filename_to_save );
    }
/*************   Example of how to predict a single sample ************************/   
// Use that for the assignment3, for every frame after computing the features, r is the prediction given the features listed in this format
    //Mat sample = data.row(i);
//     Mat sample1 = (Mat_<float>(1,9) << 1.52101, 13.64, 4.4899998, 1.1, 71.779999, 0.059999999, 8.75, 0, 0);// 1
//     float r = model->predict( sample1 );
//     cout << "Prediction: " << r << endl;
//     sample1 = (Mat_<float>(1,9) << 1.518, 13.71, 3.9300001, 1.54, 71.809998, 0.54000002, 8.21, 0, 0.15000001);//2
//     r = model->predict( sample1 );
//     cout << "Prediction: " << r << endl;
//     sample1 = (Mat_<float>(1,9) << 1.51694,12.86,3.58,1.31,72.61,0.61,8.79,0,0);//3
//     r = model->predict( sample1 );
//     cout << "Prediction: " << r << endl;
// //    sample1 = (Mat_<float>(1,9) << );//4
// //    r = model->predict( sample1 );
// //    cout << "Prediction: " << r << endl;
//     sample1 = (Mat_<float>(1,9) << 1.5151401, 14.01, 2.6800001, 3.5, 69.889999, 1.6799999, 5.8699999, 2.2, 0);//5
//     r = model->predict( sample1 );
//     cout << "Prediction: " << r << endl;
//     sample1 = (Mat_<float>(1,9) << 1.51852, 14.09, 2.1900001, 1.66, 72.669998, 0, 9.3199997, 0, 0);//6
//     r = model->predict( sample1 );
//     cout << "Prediction: " << r << endl;
//     sample1 = (Mat_<float>(1,9) << 1.51131,13.69,3.2,1.81,72.81,1.76,5.43,1.19,0);//7
//     r = model->predict( sample1 );
//     cout << "Prediction: " << r << endl;
    cout<<"problem_val: "<<problem_val<<endl;
    // getchar();
    return confusion_Matrix;
    
/**********************************************************************/
}

template<class T>//base template before specialized
class Child_ML : public Parent_ML{
public:
    // Machine_Learning *ml;
    void Intialize(){cout<<"error choose different technique"<<endl;}
    void Return_Parameter(int index){cout<<"error"<<endl;}
    string Head_Parameter(){return "error This is default child";}
    Mat *Calculate_Confusion_Matrices(int type, Machine_Learning_Data_Preparation *&i_ml){
        Mat *out_mat;
        return out_mat;
    }
    float *Calculate_Accuracies(Mat *&i_confusion_matrix, Machine_Learning_Data_Preparation *&i_ml){
        float *out_float;
        return out_float;
    }
    float Calculate_Sum_Accuracy(float *&i_accuracy, Machine_Learning_Data_Preparation *&i_ml){
        return 0;
    }
};

template<>
class Child_ML<ANN_MLP> : public Parent_ML{
public:
    Ptr<ANN_MLP> *model;
    // int t_method = ANN_MLP::BACKPROP;//default
    int t_method = ANN_MLP::BACKPROP;//default
    // int t_method = 1;//default
    int a_function = ANN_MLP::SIGMOID_SYM;//default
    int max_iter=100;
    double method_param=0.1;

    Child_ML(){}
    Child_ML(int p1, int p2, float p3, int p4, int p5){
        max_iter=p1;
        method_param=p3;
    }
    ~Child_ML(){delete model;}
    void Intialize();
    void Calculate_Result();
    Mat *Calculate_Confusion_Matrices(int type, Machine_Learning_Data_Preparation *&i_ml);
    float *Calculate_Accuracies(Mat *&i_confusion_matrix, Machine_Learning_Data_Preparation *&i_ml);
    float Calculate_Sum_Accuracy(float *&i_accuracy, Machine_Learning_Data_Preparation *&i_ml);
    void Return_Parameter(int index);
    string Head_Parameter();
};

float *Child_ML<ANN_MLP>::Calculate_Accuracies(Mat *&i_confusion_matrix, Machine_Learning_Data_Preparation *&i_ml){
    float *out_accuracies;
    out_accuracies=new float[i_ml->k_fold_value];

    for(int i=0; i< i_ml->k_fold_value; i++){
        out_accuracies[i]=Accuracy_Calculation(i_confusion_matrix[i]);
    }
    return out_accuracies;
}

Mat *Child_ML<ANN_MLP>::Calculate_Confusion_Matrices(int type, Machine_Learning_Data_Preparation *&i_ml){
    Mat *out_confusion_matrix;
    out_confusion_matrix=new Mat[i_ml->k_fold_value];

    for(int i=0; i< i_ml->k_fold_value; i++){
        if(type==0){//test
            out_confusion_matrix[i]=test_and_save_classifier(model[i], i_ml->test_data[i], i_ml->test_responses_int[i], i_ml->ntest_samples, 0, i_ml->filename_to_save, i_ml->ml_technique);
        }else{//train
            out_confusion_matrix[i]=test_and_save_classifier(model[i], i_ml->train_data[i], i_ml->train_responses_int[i], i_ml->ntrain_samples, 0, i_ml->filename_to_save, i_ml->ml_technique);
        }
        
    }
    return out_confusion_matrix;
}

float Child_ML<ANN_MLP>::Calculate_Sum_Accuracy(float *&i_accuracy, Machine_Learning_Data_Preparation *&i_ml){
    float out_sum_accuracy=0;
    for(int i=0; i< i_ml->k_fold_value; i++){
        out_sum_accuracy=out_sum_accuracy+i_accuracy[i];
        Return_Parameter(i);
    }
    return out_sum_accuracy;
}

void Child_ML<ANN_MLP>::Intialize(){
    // double p_method_param, int p_max_iter
    cout<<"start initialize ANN"<<endl;
    // cout<<"ml->k_fold_value: "<<ml->k_fold_value<<endl;
    model=new Ptr<ANN_MLP>[ml->k_fold_value];
    
    cout<<"ml->the_number_of_data: "<<ml->the_number_of_data<<endl;
    int layer_sz[] = { ml->the_number_of_data, 100, 100, ml->class_count };
    cout <<  " sizeof layer_sz " << sizeof(layer_sz) << " sizeof layer_sz[0]) " << sizeof(layer_sz[0]) << endl;
    int nlayers = (int)(sizeof(layer_sz)/sizeof(layer_sz[0]));
    cout << " nlayers  " << nlayers << endl;    
    Mat layer_sizes( 1, nlayers, CV_32S, layer_sz );
    // cout<<"ml->tdata[i]: "<<ml->tdata[0]<<endl;
    for(int i=0;i<ml->k_fold_value;i++){
        cout << "iteration ("<<i<<") Training the classifier (may take a few minutes)...\n";
        model[i] = ANN_MLP::create();
        model[i]->setLayerSizes(layer_sizes);
        model[i]->setActivationFunction(a_function, 0, 0);
        // model->setActivationFunction(ANN_MLP::IDENTITY, 0, 0);
        model[i]->setTermCriteria(ml->TC(max_iter,method_param));
        model[i]->train(ml->tdata[i]);
    }
    cout << endl;
}

void Child_ML<ANN_MLP>::Calculate_Result(){
    for(int i=0;i<ml->k_fold_value;i++){
        confusion_matrix[i]=test_and_save_classifier(model[i], ml->test_data[i], ml->test_responses_int[i], ml->ntest_samples, 0, ml->filename_to_save,ml->ml_technique);
        accuracy[i]=Accuracy_Calculation(confusion_matrix[i]);
        sum_accuracy=sum_accuracy+accuracy[i];
        Return_Parameter(i);
    }
}

void Child_ML<ANN_MLP>::Return_Parameter(int index ){
    sprintf(result_buffer[index], "%d, %d, %d, %lf, %d, %d, %f \n", index, t_method, a_function, method_param, max_iter, ml->class_count, accuracy[index]);  //header
}

string Child_ML<ANN_MLP>::Head_Parameter(){
    return "Index, Method_Type, a_function, MethodParameter, MaxIteration, ClassCount, Accuracy";
}

template<>


class Child_ML<Boost> : public Parent_ML{

private:
    int boost_type=0;//Gentle 0.5 and true{DISCRETE, REAL, LOGIT, GENTLE}
    int weak_count=100;
    float weight_trim_rate=80.83;
    int max_depth=10;
public:
    Ptr<Boost> *model;
    Child_ML(){}
    Child_ML(int p1, int p2, float p3, int p4, int p5){
        boost_type=p1;//Gentle 0.5 and true{DISCRETE, REAL, LOGIT, GENTLE}
        weak_count=p2;
        weight_trim_rate=p3;
        max_depth=p4;
    }
    ~Child_ML(){delete model;}
    void Intialize();
    void Calculate_Result();
    Mat *Calculate_Confusion_Matrices(int type, Machine_Learning_Data_Preparation *&i_ml);
    float *Calculate_Accuracies(Mat *&i_confusion_matrix, Machine_Learning_Data_Preparation *&i_ml);
    float Calculate_Sum_Accuracy(float *&i_accuracy, Machine_Learning_Data_Preparation *&i_ml);
    void Return_Parameter(int index);
    string Head_Parameter();
};
void Child_ML<Boost> ::Intialize(){
    model=new Ptr<Boost>[ml->k_fold_value];
    for(int i=0;i<ml->k_fold_value;i++){
        cout << "iteration ("<<i<<") Training the classifier (may take a few minutes)...\n";
        model[i]=Boost::create();
        model[i]->setBoostType(boost_type);  //Gentle 0.5 and true{DISCRETE, REAL, LOGIT, GENTLE}
        model[i]->setWeakCount(weak_count);       //the Gentle best=98;
        model[i]->setWeightTrimRate(weight_trim_rate);//the Gentle best=0.83;
        model[i]->setMaxDepth(max_depth);         //the Gentle best=2;
        model[i]->setUseSurrogates(false);
        model[i]->setPriors(Mat()); 
        cout << "Training the classifier (may take a few minutes)...\n";
        model[i]->train(ml->tdata[i]);
    }
    cout << endl;
}


Mat *Child_ML<Boost>::Calculate_Confusion_Matrices(int type, Machine_Learning_Data_Preparation *&i_ml){
    Mat *out_confusion_matrix;
    out_confusion_matrix=new Mat[i_ml->k_fold_value];

    for(int i=0; i< i_ml->k_fold_value; i++){
        if(type==0){//test
            out_confusion_matrix[i]=test_and_save_classifier(model[i], i_ml->test_data[i], i_ml->test_responses_int[i], i_ml->ntest_samples, 0, i_ml->filename_to_save, i_ml->ml_technique);
        }else{//train
            out_confusion_matrix[i]=test_and_save_classifier(model[i], i_ml->train_data[i], i_ml->train_responses_int[i], i_ml->ntrain_samples, 0, i_ml->filename_to_save, i_ml->ml_technique);
        }
    }
    return out_confusion_matrix;
}

float *Child_ML<Boost>::Calculate_Accuracies(Mat *&i_confusion_matrix, Machine_Learning_Data_Preparation *&i_ml){
    float *out_accuracies;
    out_accuracies=new float[i_ml->k_fold_value];

    for(int i=0; i< i_ml->k_fold_value; i++){
        out_accuracies[i]=Accuracy_Calculation(i_confusion_matrix[i]);
    }
    return out_accuracies;
}

float Child_ML<Boost>::Calculate_Sum_Accuracy(float *&i_accuracy, Machine_Learning_Data_Preparation *&i_ml){
    float out_sum_accuracy=0;
    for(int i=0; i< i_ml->k_fold_value; i++){
        out_sum_accuracy=out_sum_accuracy+i_accuracy[i];
        Return_Parameter(i);
    }
    return out_sum_accuracy;
}

void Child_ML<Boost> ::Calculate_Result(){
    for(int i=0;i<ml->k_fold_value;i++){
        confusion_matrix[i]=test_and_save_classifier(model[i], ml->test_data[i], ml->test_responses_int[i], ml->ntest_samples, 0, ml->filename_to_save,ml->ml_technique);
        accuracy[i]=Accuracy_Calculation(confusion_matrix[i]);
        sum_accuracy=sum_accuracy+accuracy[i];
        Return_Parameter(i);
    }
}


void Child_ML<Boost>::Return_Parameter(int index){
    sprintf(result_buffer[index], "%d, %d, %d, %f, %d, %d, %f \n", index, boost_type, weak_count, weight_trim_rate, max_depth, ml->class_count, accuracy[index]);  //header
}

string Child_ML<Boost>::Head_Parameter(){
    return "Index, BoostType, WeakCount, WeightTrimRate, MaxDepth, ClassCount, Accuracy";
}


template<>
class Child_ML<RTrees> : public Parent_ML{

private:
    int max_depth=12;
    int min_sample_count=5;
    float regression_accuracy=0.01f;
    int max_categories=2;
    int tc_value=100;

public:
    Ptr<RTrees> *model;
    Child_ML(){}
    Child_ML(int p1, int p2, float p3, int p4, int p5){
        max_depth=p1;
        min_sample_count=p2;
        regression_accuracy=p3;
        max_categories=p4;
        tc_value=p5;
    }
    ~Child_ML(){delete model;}
    void Intialize();
    void Calculate_Result();
    Mat *Calculate_Confusion_Matrices(int type, Machine_Learning_Data_Preparation *&i_ml);
    float *Calculate_Accuracies(Mat *&i_confusion_matrix, Machine_Learning_Data_Preparation *&i_ml);
    float Calculate_Sum_Accuracy(float *&i_accuracy, Machine_Learning_Data_Preparation *&i_ml);
    void Return_Parameter(int index);
    string Head_Parameter();
};

void Child_ML<RTrees>::Intialize(){
    sprintf(final_result_buffer, "%d, %d, %f, %d, %d", max_depth, min_sample_count, regression_accuracy, tc_value,ml->class_count);  //header
    // cout<<"THis is RF"<<endl;
    // getchar();
    
    model=new Ptr<RTrees>[ml->k_fold_value];
    
    for(int i=0;i<ml->k_fold_value;i++){
        model[i] = RTrees::create();
        model[i]->setMaxDepth(max_depth);
        model[i]->setMinSampleCount(min_sample_count);
        model[i]->setRegressionAccuracy(regression_accuracy);
        model[i]->setUseSurrogates(false /* true */);
        model[i]->setMaxCategories(2);
        model[i]->setTermCriteria(ml->TC(tc_value,0));
        cout << "Training the classifier (may take a few minutes)...\n";
        // model[i]->setTermCriteria(TermCriteria(TermCriteria::COUNT, 50, 0));
        model[i]->train(ml->tdata[i]);
    }
    // cout<<"tc_value: "<<tc_value<<endl;
    // getchar();
    cout << endl;
}

float *Child_ML<RTrees>::Calculate_Accuracies(Mat *&i_confusion_matrix, Machine_Learning_Data_Preparation *&i_ml){
    float *out_accuracies;
    out_accuracies=new float[i_ml->k_fold_value];

    for(int i=0; i< i_ml->k_fold_value; i++){
        out_accuracies[i]=Accuracy_Calculation(i_confusion_matrix[i]);
    }
    return out_accuracies;
}

Mat *Child_ML<RTrees>::Calculate_Confusion_Matrices(int type, Machine_Learning_Data_Preparation *&i_ml){
    Mat *out_confusion_matrix;
    out_confusion_matrix=new Mat[i_ml->k_fold_value];

    for(int i=0; i< i_ml->k_fold_value; i++){
        if(type==0){//test
            out_confusion_matrix[i]=test_and_save_classifier(model[i], i_ml->test_data[i], i_ml->test_responses_int[i], i_ml->ntest_samples, 0, i_ml->filename_to_save, i_ml->ml_technique);
        }else{//train
            out_confusion_matrix[i]=test_and_save_classifier(model[i], i_ml->train_data[i], i_ml->train_responses_int[i], i_ml->ntrain_samples, 0, i_ml->filename_to_save, i_ml->ml_technique);
        }
    }
    return out_confusion_matrix;
}

float Child_ML<RTrees>::Calculate_Sum_Accuracy(float *&i_accuracy, Machine_Learning_Data_Preparation *&i_ml){
    float out_sum_accuracy=0;
    for(int i=0; i< i_ml->k_fold_value; i++){
        out_sum_accuracy=out_sum_accuracy+i_accuracy[i];
        Return_Parameter(i);
    }
    return out_sum_accuracy;
}

void Child_ML<RTrees>::Calculate_Result(){
    for(int i=0;i<ml->k_fold_value;i++){
        confusion_matrix[i]=test_and_save_classifier(model[i], ml->test_data[i], ml->test_responses_int[i], ml->ntest_samples, 0, ml->filename_to_save,ml->ml_technique);
        // confusion_matrix[i]=test_and_save_classifier(model[i], ml->train_data[i], ml->train_responses_int[i], ml->ntrain_samples, 0, ml->filename_to_save,ml->ml_technique);
        accuracy[i]=Accuracy_Calculation(confusion_matrix[i]);
        sum_accuracy=sum_accuracy+accuracy[i];
        Return_Parameter(i);
    }
}
void Child_ML<RTrees>::Return_Parameter(int index){
    sprintf(result_buffer[index], "%d, %d, %d, %f, %d, %d, %f \n", index, max_depth, min_sample_count, regression_accuracy, tc_value,ml->class_count, accuracy[index]);  //header
    
}

string Child_ML<RTrees>::Head_Parameter(){
    return "Index, MaxDepth, RegressionAccuracy, MaxCategories, TermCritera, ClassCount, Accuracy";
}

template<>
class Child_ML<SVM> : public Parent_ML{

private:
    // int max_depth=12;
    // int min_sample_count=5;
    // float regression_accuracy=0.01f;
    // int max_categories=2;
    // int tc_value=100;

public:
    Ptr<SVM> *model;
    Child_ML(){}
    Child_ML(int p1, int p2, float p3, int p4, int p5){
        // max_depth=p1;
        // min_sample_count=p2;
        // regression_accuracy=p3;
        // max_categories=p4;
        // tc_value=p5;
    }
    ~Child_ML(){delete model;}
    void Intialize();
    Mat *Calculate_Confusion_Matrices(int type, Machine_Learning_Data_Preparation *&i_ml);
    float *Calculate_Accuracies(Mat *&i_confusion_matrix, Machine_Learning_Data_Preparation *&i_ml);
    float Calculate_Sum_Accuracy(float *&i_accuracy, Machine_Learning_Data_Preparation *&i_ml);
    void Calculate_Result();
    void Return_Parameter(int index);
    string Head_Parameter();
};

void Child_ML<SVM>::Intialize(){
    // sprintf(final_result_buffer, "%d, %d, %f, %d, %d", max_depth, min_sample_count, regression_accuracy, tc_value,ml->class_count);  //header
    // cout<<"THis is RF"<<endl;
    // getchar();
    
    model=new Ptr<SVM>[ml->k_fold_value];
    
    for(int i=0;i<ml->k_fold_value;i++){
        model[i] = SVM::create();
        model[i]->setType(SVM::EPS_SVR);//C_SVC, NU_SVC, ONE_CLASS, EPS_SVR, NU_SVR
        model[i]->setKernel(SVM::RBF);//CUSTOM, LINEAR, POLY, RBF, SIGMOID, CHI2, INTER
        model[i]->setDegree(0);
        model[i]->setGamma(10);
        model[i]->setCoef0(0.0);
        model[i]->setC(5);
        model[i]->setNu(0);
        model[i]->setP(10);
        model[i]->setClassWeights(Mat());
        model[i]->setTermCriteria(ml->TC(100,0));
        // model[i]->setTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001);
        cout << "Training the classifier (may take a few minutes)...\n";
        // model[i]->setTermCriteria(TermCriteria(TermCriteria::COUNT, 50, 0));
        model[i]->train(ml->tdata[i]);
    }
    // cout<<"tc_value: "<<tc_value<<endl;
    // getchar();
    cout << endl;
}

float *Child_ML<SVM>::Calculate_Accuracies(Mat *&i_confusion_matrix, Machine_Learning_Data_Preparation *&i_ml){
    float *out_accuracies;
    out_accuracies=new float[i_ml->k_fold_value];

    for(int i=0; i< i_ml->k_fold_value; i++){
        out_accuracies[i]=Accuracy_Calculation(i_confusion_matrix[i]);
    }
    return out_accuracies;
}

Mat *Child_ML<SVM>::Calculate_Confusion_Matrices(int type, Machine_Learning_Data_Preparation *&i_ml){
    Mat *out_confusion_matrix;
    out_confusion_matrix=new Mat[i_ml->k_fold_value];
    for(int i=0; i< i_ml->k_fold_value; i++){
        if(type==0){//test
            out_confusion_matrix[i]=test_and_save_classifier(model[i], i_ml->test_data[i], i_ml->test_responses_int[i], i_ml->ntest_samples, 0, i_ml->filename_to_save, i_ml->ml_technique);
        }else{//train
            out_confusion_matrix[i]=test_and_save_classifier(model[i], i_ml->train_data[i], i_ml->train_responses_int[i], i_ml->ntrain_samples, 0, i_ml->filename_to_save, i_ml->ml_technique);
        }
    }
    return out_confusion_matrix;
}

float Child_ML<SVM>::Calculate_Sum_Accuracy(float *&i_accuracy, Machine_Learning_Data_Preparation *&i_ml){
    float out_sum_accuracy=0;
    for(int i=0; i< i_ml->k_fold_value; i++){
        out_sum_accuracy=out_sum_accuracy+i_accuracy[i];
        Return_Parameter(i);
    }
    return out_sum_accuracy;
}

void Child_ML<SVM>::Calculate_Result(){
    for(int i=0;i<ml->k_fold_value;i++){
        confusion_matrix[i]=test_and_save_classifier(model[i], ml->test_data[i], ml->test_responses_int[i], ml->ntest_samples, 0, ml->filename_to_save,ml->ml_technique);
        // confusion_matrix[i]=test_and_save_classifier(model[i], ml->train_data[i], ml->train_responses_int[i], ml->ntrain_samples, 0, ml->filename_to_save,ml->ml_technique);
        accuracy[i]=Accuracy_Calculation(confusion_matrix[i]);
        sum_accuracy=sum_accuracy+accuracy[i];
        Return_Parameter(i);
    }
}
void Child_ML<SVM>::Return_Parameter(int index){
    // sprintf(result_buffer[index], "%d, %d, %d, %f, %d, %d, %f \n", index, max_depth, min_sample_count, regression_accuracy, tc_value,ml->class_count, accuracy[index]);  //header
    
}

string Child_ML<SVM>::Head_Parameter(){
    return "Index, MaxDepth, RegressionAccuracy, MaxCategories, TermCritera, ClassCount, Accuracy";
}


class Write_File{
private:
    Parent_ML *final_ml;
    Machine_Learning_Data_Preparation *prepared_data;
    float mean;
    float variance; 
    float sta_dev; 
    int k_fold_value; 
    Mat *con_mat;
    char **buffer_file;
    string file_full_path;
    string file_the_best_full_path;
    string file_collection_full_path;
    ofstream file;
    ofstream file_the_best;
    ofstream file_collection;
public:
    Write_File();
    Write_File(Parent_ML *i_final_ml, Machine_Learning_Data_Preparation *i_prepared_data,string i_number_of_CE);
    ~Write_File();
    // void Main_Process(float mean, float variance,float sta_dev,int k_fold_value, Mat con_mat[],char **buffer_file);
    void Main_Process();
    void Write_Header();
    string Create_file_path(string file_path, string file_name, string number_of_CE);
    bool The_File_Process();
    bool The_File_Collection_Process(float i_mean);
    bool The_Best_Process(float i_mean, float i_variance, float i_sta_dev, int i_k_fold_value,Mat *&i_con_mat, string type);
};

Write_File::Write_File(Parent_ML *i_final_ml, Machine_Learning_Data_Preparation *i_prepared_data,string i_number_of_CE){
    final_ml=i_final_ml;
    prepared_data=i_prepared_data;
    // mean=final_ml->mean;
    // variance=final_ml->variance; 
    // sta_dev=final_ml->sta_dev;
    // k_fold_value=prepared_data->k_fold_value;
    // con_mat=final_ml->confusion_matrix;
    buffer_file=final_ml->result_buffer;

    string file_path="resource/rf/";
    file_full_path=Create_file_path(file_path,"min_sample_count_",i_number_of_CE);
    file_the_best_full_path=Create_file_path(file_path,"Calculate_standard_deviation_",i_number_of_CE);
    file_collection_full_path=Create_file_path(file_path,"accuracy_collection",i_number_of_CE);

    file.open(file_full_path, std::ios_base::app);
    file_collection.open(file_collection_full_path, std::ios_base::app);
    file_the_best.open(file_the_best_full_path, std::ios_base::app);
}

Write_File::~Write_File(){
    file.close();
    file_the_best.close();
    file_collection.close();
}
void Write_File::Write_Header(){
    file_collection<<"#MaxDepth, RegressionAccuracy, MaxCategories, TermCritera, ClassCount, Accuracy";
    file_collection<<endl;
    file<<final_ml->Head_Parameter();
    file<<endl;
}
void Write_File::Main_Process(){
    // The_Best_Process();
    // The_File_Collection_Process(final_ml->mean_test);
    // The_File_Process();


    The_Best_Process(final_ml->mean_test, final_ml->variance_test, final_ml->sta_dev_test, 
        prepared_data->k_fold_value, final_ml->confusion_matrix_test, "test");
    The_Best_Process(final_ml->mean_train, final_ml->variance_train, final_ml->sta_dev_train, 
        prepared_data->k_fold_value, final_ml->confusion_matrix_train, "train");

}

string Write_File::Create_file_path(string file_path, string file_name, string number_of_CE){
    string file_name_extension=".txt";
    string full_file_name=file_name+number_of_CE+file_name_extension;
    string temp_file_full_path=file_path+full_file_name;
    return temp_file_full_path;
}

bool Write_File::The_File_Process(){
    for(int i=0;i<k_fold_value;i++){
        file<<buffer_file[i];
    }
    return true;        
}

bool Write_File::The_File_Collection_Process(float i_mean){
    // char mse_buffer[70];
    // sprintf(mse_buffer, "%1.f ± %1.f%% \n", mean*100,sta_dev*100);
    // file_collection<<"#Mean, Variance, Sta_dev, Mean Square Error";
    // file_collection<<endl;
    // file_collection<<to_string(mean);
    // file_collection<<", ";
    // file_collection<<to_string(variance);
    // file_collection<<", ";
    // file_collection<<to_string(sta_dev);
    // file_collection<<", ";
    // file_collection<<mse_buffer;
    // file_collection<<endl;
    file_collection<<final_ml->final_result_buffer;
    file_collection<<", ";
    file_collection<<to_string(i_mean);
    file_collection<<endl; 
    
    return true;
}

bool Write_File::The_Best_Process(float i_mean, float i_variance, float i_sta_dev, int i_k_fold_value, Mat *&i_con_mat, string type){
    char mean_buffer[20],variance_buffer[40],sta_dev_buffer[40],mse_buffer[70];
    sprintf(mean_buffer, "#mean: %f \n", i_mean);
    sprintf(variance_buffer, "#variance: %f \n", i_variance);
    sprintf(sta_dev_buffer, "#sta_dev: %f \n", i_sta_dev);  //header
    sprintf(mse_buffer, "#Mean Square Error: %1.f ± %1.f%% \n", i_mean*100, i_sta_dev*100);
    // cout<<"start Best"<<endl;
    // cout<<"mean:"<<mean_buffer<<endl;
    // cout<<"variance_buffer:"<<variance_buffer<<endl;
    // cout<<"sta_dev_buffer:"<<sta_dev_buffer<<endl;
    // cout<<"mse_buffer:"<<mse_buffer<<endl;
    // cout<<"end Best"<<endl;
    file_the_best << "----------";
    file_the_best << type;
    file_the_best << "----------\n";
    if (file_the_best){
        // file_the_best<<"\n\n";    
        file_the_best<<mean_buffer;
        file_the_best<<variance_buffer;
        file_the_best<<sta_dev_buffer;
        file_the_best<<mse_buffer;
        file_the_best<<"\n\n";
        file_the_best<<"#Confusion Matrix\n";
        for(int i=0; i<i_k_fold_value; i++){
            char buffer[50];
            sprintf(buffer, "#k=%d\n", i);  //header
            file_the_best<<buffer;
            file_the_best<<"#";
            file_the_best<<i_con_mat[i].at<int>(0,0);
            file_the_best<<", ";
            file_the_best<<i_con_mat[i].at<int>(0,1);
            file_the_best<<"\n";
            file_the_best<<"#";
            file_the_best<<i_con_mat[i].at<int>(1,0);
            file_the_best<<", ";
            file_the_best<<i_con_mat[i].at<int>(1,1);
            file_the_best<<"\n\n";
        }
            
    }
    file_the_best << "--------";
    file_the_best << type;
    file_the_best << " end";
    file_the_best << "--------\n";
    return 0;
}
#endif // end of LOAD_AND_SAVE_ML_H