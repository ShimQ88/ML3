// Project Headers
#include "load_and_save_ml.h"

int
Count_Column_Numb(const string& filename){//only two case sample work need to develop more
    int numb_of_data_cols;
    ifstream myfile(filename);
    ofstream shuffled_file("data/shuffle_output.txt");
    string line1[3000];
    string line2[3000];
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
                // if(temp_line[0]=='1')
            {
                // cout<<"hahaha"<<endl;
                line1[i]=temp_line;
                i++;
            }
            else if(temp_line[0]=='1')
                // else if(temp_line[0]=='9')
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

    // Mat el_ptr(1, var_count, CV_32F);
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
        cout << "responses " << buf[0] << " " <<endl;//<<  endl;
        ptr = buf+2;
        for( i = 0; i < var_count; i++ )//in case of name is included
            // for( i = 0; i < var_count; i++ )//in case of name is excluded
        {
            int n = 0;
            sscanf( ptr, "%f%n", &el_ptr.at<float>(i), &n );
            ptr += n + 1;
            // cout<<"ptr: "<<ptr<<endl;
        }
        cout << el_ptr << endl;
        // getchar();
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
void Test(string input_file_name, string output_file_name){
    ifstream input_file(input_file_name);
    ofstream output_file(output_file_name);
    string read_line;

    int delimiter=0;
    getline(input_file,read_line);
    if((input_file.is_open())&&(output_file.is_open())){
        while(getline(input_file, read_line)){
            // strinh output_string;
            int delimiter=0;
            string temp_read_line=read_line;//copy string

            while(delimiter!=-1){
                delimiter = temp_read_line.find(',');
                string splited_value=temp_read_line.substr(0,delimiter);
                // cout<<"splited_value: "<<splited_value<<endl;
                // cout<<"splited_value.length(): "<<splited_value.length()<<endl;
                char str[splited_value.length()];
                bool is_digit=true;
                for (int i=0; i<splited_value.length(); i++){
                    str[i]=splited_value[i];
                    // cout<<"str[i]:"<<str[i]<<endl;
                    // getchar();
                    if((isdigit(str[i])==true)||(str[i]=='.')){
                        
                    }else{
                        cout<<"this is not digit"<<endl;
                        is_digit=false;
                        break;
                    }
                }
                if(is_digit==true){
                    if(delimiter==-1){
                        output_file<<splited_value;
                    }else{
                        output_file<<splited_value;
                        output_file<<',';    
                    }
                    
                }else{
                    // output_file<<splited_value;
                    // output_file<<',';
                    //doing nothing
                }
                temp_read_line = temp_read_line.substr(delimiter+1);//delete copied data
            }
            output_file<<"\n";
        }
    }

    input_file.close();
    output_file.close();
}

void Test2(string input_file_name, string output_file_name){
    ifstream input_file(input_file_name);
    ofstream output_file(output_file_name);
    string read_line;

    int delimiter=0;
    getline(input_file,read_line);
    if((input_file.is_open())&&(output_file.is_open())){
        while(getline(input_file, read_line)){
            // strinh output_string;
            int delimiter=0;
            string temp_read_line=read_line;//copy string
            int k=0;
            while(delimiter!=-1){
                delimiter = temp_read_line.find(',');
                string splited_value=temp_read_line.substr(0,delimiter);
                // cout<<"splited_value: "<<splited_value<<endl;
                // cout<<"splited_value.length(): "<<splited_value.length()<<endl;
                char str[splited_value.length()];
                bool is_digit=true;
                for (int i=0; i<splited_value.length(); i++){
                    str[i]=splited_value[i];
                    // cout<<"str[i]:"<<str[i]<<endl;
                    // getchar();
                    if((isdigit(str[i])==true)||(str[i]=='.')){
                        
                    }else{
                        cout<<"this is not digit"<<endl;
                        is_digit=false;
                        break;
                    }
                }
                // cout<<"k: "<<k<<endl;
                // getchar();
                if(is_digit==true){
                    if(delimiter==-1){
                        output_file<<splited_value;
                    }else{
                        if(k==0){
                            output_file<<"9,";
                            // cout<<"ha"<<endl;
                            // getchar();
                            // output_file<<',';    
                        }else{
                            output_file<<splited_value;
                            output_file<<',';    
                        }
                        
                    }
                    
                }else{
                    output_file<<splited_value;
                    output_file<<',';
                    // doing nothing
                }
                k++;
                temp_read_line = temp_read_line.substr(delimiter+1);//delete copied data
            }
            output_file<<"\n";
        }
    }

    input_file.close();
    output_file.close();
}

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
    // string data_filename2="Final_dataset/contour_name_removed.data";
    Mat data;
    Mat responses;
    int numb_of_data_cols=Count_Column_Numb(data_filename);
    string name="data/shuffle_output.txt";
    string name2="data/shuffle_output_name_removed.txt";
    Test(name,name2);


    // Test2("Final_dataset/temp/contour_bird.txt","Final_dataset/temp/contour_bird_final.txt");
    // Test2("Final_dataset/temp/contour_rodent.txt","Final_dataset/temp/contour_rodent_final.txt");
    // exit(1);
    // cout<<"numb_of_data_cols: "<<numb_of_data_cols<<endl;
    // getchar();

    bool ok = read_num_class_data( name2, numb_of_data_cols-1, &data, &responses );//third parameter: FEATURES
    // bool ok = read_num_class_data( name, 10, &data, &responses );//third parameter: FEATURES
    // bool ok = read_num_class_data( data_filename, numb_of_data_cols, &data, &responses );//third parameter: FEATURES
    if( !ok ){
        cout<<"error from read file"<<endl;
        return ok;
    }
    cout<<"responses: "<<responses<<endl;
    cout<<"data: "<<data<<endl;
    
    //preparing part
    int nsamples_all = data.rows;
    cout<<"nsamples_all: "<<nsamples_all<<endl;
    getchar();
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

    

    
    // ofstream acc;
    // acc.open("resource/rf/accuracy_collection.txt");
    // acc<<"i, MD, MSC, NT, MSE(index, setMaxDepth, setMinSampleCount, Number_of_Trees, Mean_Sqare_Error)\n";
    
    
    // int min_samp_count=5;
    // int i=0;
    // while(1){
    //     int max_dep=6;
    //     while(1){
    //         int TC=10000;
    //         while(1){
    //             Parent_ML *final_ml;
    //             if(ml_technique==0){
    //                 final_ml = Creat_ML_Class< Child_ML<ANN_MLP> >();  
    //             }else if(ml_technique==1){
    //                 final_ml = Creat_ML_Class< Child_ML<Boost> >();
    //             }else if(ml_technique==2){
    //                 final_ml = Creat_ML_Class< Child_ML<RTrees> >();
    //             }else{
    //                 cout<<"ml_technique code error"<<endl;
    //                 return false;
    //             }

    //             final_ml->Main_Process(prepared_data,max_dep,min_samp_count,TC);//doing main process
                
    //             // string numb_ce=to_string(prepared_data->the_number_of_data+1);//check number of CEs
    //             string numb_ce=to_string(i);//check number of CEs
                
    //             Write_File file_write(numb_ce);//writing file class
    //             // Write_File file_write(numb_ce);//writing file class
    //             // cout<<"seg"<<endl;
    //             // cout<<"final_ml->variance: "<<final_ml->variance<<endl;
    //             file_write.Main_Process(final_ml->mean, final_ml->variance, final_ml->sta_dev,prepared_data->k_fold_value,final_ml->confusion_matrix, final_ml->result_buffer);
                                
    //             acc<<to_string(i);
    //             acc<<", ";
    //             acc<<to_string(max_dep);
    //             acc<<", ";
    //             acc<<to_string(min_samp_count);
    //             acc<<", ";
    //             acc<<to_string(TC);
    //             acc<<", ";
    //             acc<<to_string(final_ml->mean);
    //             acc<<"\n";
    //             i++;
    //             if(TC>=10000){break;}
    //             TC=TC+1000;
    //         }
    //         max_dep=max_dep+1;
    //         if(max_dep>6){break;}
    //     }
    //     min_samp_count=min_samp_count+1;
    //     if(min_samp_count>=5){break;}
    // }
    // acc.close();
    // cout<<"eh"<<endl;
    // delete final_ml;
    
    int i=0;
    Machine_Learning_Data_Preparation *prepared_data = new Machine_Learning_Data_Preparation(data, responses, ntrain_samples,ntest_samples, filename_to_save, filename_to_load, 2);
    prepared_data->Main_Process(ml_technique);//data arrangement
    
    //parameters in ANN
    int max_iter=100;
    float method_param=0.01;
    //parameters in Boost
    int boost_type=0;//Gentle 0.5 and true{DISCRETE, REAL, LOGIT, GENTLE}
    int weak_count=100;
    float weight_trim_rate=80.83;
    int max_depth=12;
    //parameters in Random Forest Trees
    //int max_depth=10;
    int min_sample_count=5;
    float regression_accuracy=0.01f;
    int max_categories=2;
    int tc_value=10;

    Parent_ML *final_ml;
    if(ml_technique==0){
        final_ml = Creat_ML_Class< Child_ML <ANN_MLP> >(max_iter,0,method_param,0,0);//{max_iter,null,method_param,null,null}
    }else if(ml_technique==1){
        final_ml = Creat_ML_Class< Child_ML <Boost> >(boost_type,weak_count,weight_trim_rate,max_depth,0);
    }else if(ml_technique==2){
        final_ml = Creat_ML_Class< Child_ML <RTrees> >(max_depth,min_sample_count,regression_accuracy,max_categories,tc_value);
    }else{
        cout<<"ml_technique code error"<<endl;
        return false;
    }

    final_ml->Main_Process(prepared_data);//doing main process
    

    // string numb_ce=to_string(prepared_data->the_number_of_data+1);//check number of CEs
    // string numb_ce=to_string(i);//check number of CEs
    
    Write_File file_write(final_ml,prepared_data,"0");//writing file class
    // Write_File file_write(numb_ce);//writing file class
    // cout<<"seg"<<endl;
    // cout<<"final_ml->variance: "<<final_ml->variance<<endl;
    file_write.Main_Process();
                    
    // acc<<to_string(i);
    // acc<<", ";
    // acc<<to_string(max_dep);
    // acc<<", ";
    // acc<<to_string(min_samp_count);
    // acc<<", ";
    // acc<<to_string(TC);
    // acc<<", ";
    // acc<<to_string(final_ml->mean);
    // acc<<"\n";
    // i++;
    // if(TC>=10000){break;}
    // TC=TC+1000;
    // delete prepared_data;

    return true;
}