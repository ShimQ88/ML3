// Project Headers
#include "load_and_save_ml.h"

int
Count_Column_Numb(const string& filename){//only two case sample work need to develop more
    int numb_of_data_cols;
    ifstream myfile(filename);
    ofstream shuffled_file("data/shuffle_output.txt");
    string line1[300];
    string line2[300];
    string temp_line;
    int i=0;
    int j=0;
    if (myfile.is_open())
    {
        // getline (myfile,line);  
        while ( getline (myfile,temp_line) )
        {
            if(i==0){
                numb_of_data_cols=count(temp_line.begin(), temp_line.end(), ',');    
            }
            
            if(temp_line[0]=='0')
            {
                // cout<<"hahaha"<<endl;
                line1[i]=temp_line;
                i++;
            }
            else if(temp_line[0]=='1')
            {
                line2[j]=temp_line;
                j++;
            }
            else
            {
                cout<<"error"<<endl;
                getchar();
            }
            
        }
        
        // cout<<"numb_of_data_cols: "<<numb_of_data_cols<<endl;
        
        // cout<<"The first string: "<<temp_line[0];
        // getchar();
        myfile.close();
    }
    int i1=0;
    int j1=0;
    if (shuffled_file.is_open())
    {
        while(1){
            if( (i1==i)&&(j1==j) ){
                break;
            }
            if(i1!=i){
                shuffled_file <<line1[i1]+'\n';
                i1++;
            }
            if(j1!=j){
                shuffled_file << line2[j1]+'\n';
                j1++;
            }
            
        }
        shuffled_file.close();
    }
    
    return numb_of_data_cols;
}
bool
read_num_class_data( const string& filename, int var_count,
                     Mat* _data, Mat* _responses )
{
    const int M = 1024;
    char buf[M+2];

    Mat el_ptr(1, var_count, CV_32F);
    int i;
    vector<int> responses;

    _data->release();
    _responses->release();

    FILE* f = fopen( filename.c_str(), "rt" );
    if( !f )
    {
        cout << "Could not read the database " << filename << endl;
        return false;
    }

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        responses.push_back(buf[0]);
        //char test;
        //test=buf[0]+65;
        //responses.push_back(test);
        cout << "responses " << buf[0] << " " ;;//<<  endl;
        ptr = buf+2;
        for( i = 0; i < var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", &el_ptr.at<float>(i), &n );
            ptr += n + 1;
        }
        cout << el_ptr << endl;
        if( i < var_count )
            break;
        _data->push_back(el_ptr);
    }
    fclose(f);
    Mat(responses).copyTo(*_responses);

    cout << "The database " << filename << " is loaded.\n";

    return true;
}

void help(int ntrain_samples, int ntest_samples,int ml_technique){
    cout<<endl<<endl<<endl<<endl<<endl;
    cout<<"#########################################################"<<endl;
    if(ml_technique==0){
        cout<<"######## machine learning technique: Neural Network. ####"<<endl;

    }else if(ml_technique==1){
        cout<<"######## machine learning technique: Ada Boost. #########"<<endl;
    }else if(ml_technique==2){
        cout<<"######## machine learning technique: Random Forest. #####"<<endl;
    }

    cout<<"######## mode: train by training sample.        ########"<<endl;
    cout<<"########       test by testing sample.           ########"<<endl;
    printf("######## Training the index number from 0 to %d ########\n",ntrain_samples);
    printf("######## Testing the index number from %d to %d ########\n",ntrain_samples+1, ntrain_samples+ntest_samples);
    cout<<"#########################################################"<<endl;
    cout<<"Press any key"<<endl;
}


//main_ml(){}
bool
load_and_save_ml( const string& data_filename,
                      const string& filename_to_save,
                      const string& filename_to_load,
                      float percent_of_division,
                      int ml_technique)
{
    /*infomation 
       ml_technique= 1.neural_network 2.ada_boost 3.random_forest 
    */
    Mat data;
    Mat responses;
    int numb_of_data_cols=Count_Column_Numb(data_filename);
    string name="data/shuffle_output.txt";
    // cout<<"numb_of_data_cols: "<<numb_of_data_cols<<endl;
    // getchar();

    bool ok = read_num_class_data( name, numb_of_data_cols, &data, &responses );//third parameter: FEATURES
    // bool ok = read_num_class_data( data_filename, numb_of_data_cols, &data, &responses );//third parameter: FEATURES
    if( !ok ){
        cout<<"error from read file"<<endl;
        return ok;
    }

    //preparing part
    int nsamples_all = data.rows;

    /*Division part*/
    int ntrain_samples = (int)round(nsamples_all*percent_of_division);//SPLIT
    int ntest_samples = (int)round(nsamples_all*(1-percent_of_division));//SPLIT

    //this process for dividing exactly with test value
    int remainder=ntrain_samples%ntest_samples;
    ntrain_samples = ntrain_samples-remainder;

    // cout<<"ntrain_samples: "<<ntrain_samples<<endl;
    // cout<<"ntest_samples: "<<ntest_samples<<endl;


    //Print Test number of samples    
    help(ntrain_samples,ntest_samples,ml_technique);
    // getchar();
    // cout<<"responses: "<<responses<<endl;
    // cout<<"filename_to_save: "<<filename_to_save<<endl;

    

    Machine_Learning_Data_Preparation *prepared_data = new Machine_Learning_Data_Preparation(data, responses, ntrain_samples,ntest_samples, filename_to_save, filename_to_load);
    prepared_data->Main_Process(ml_technique);//data arrangement
    
    Parent_ML *final_ml;  
    if(ml_technique==0){
        final_ml = Creat_ML_Class< Child_ML<ANN_MLP> >();  
    }else if(ml_technique==1){
        final_ml = Creat_ML_Class< Child_ML<Boost> >();
    }else if(ml_technique==2){
        final_ml = Creat_ML_Class< Child_ML<RTrees> >();
    }else{
        cout<<"ml_technique code error"<<endl;
        return false;
    }

    final_ml->Main_Process(prepared_data);//doing main process
    
    string numb_ce=to_string(prepared_data->the_number_of_data+1);//check number of CEs
    
    Write_File file_write(numb_ce);//writing file class
    file_write.Main_Process(final_ml->mean, final_ml->variance, final_ml->sta_dev,prepared_data->k_fold_value,final_ml->confusion_matrix, final_ml->result_buffer);
    
    return true;
}