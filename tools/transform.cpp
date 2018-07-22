#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <stdio.h>
using namespace std;
using namespace cv;
int main(void){
    char buff1[100];
    char buff2[100];
    fstream file;
    int number = 0;
    for(int i=1;i<=139;i++)
    {
        sprintf(buff1,"/home/taylor/mirror/data/test/mask/%d_json/label.png",i);
//        cout<<buff1<<endl;
        file.open(buff1, ios::in);
        if(!file)
        {
            number += 1;
            cout<<i<<endl;
            file.close();
            continue;
        }
        else
        {
            sprintf(buff2,"/home/taylor/mirror/data/test/mask/%d_json/label8.png",i);
            //sprintf(buff1,"/media/lj/FA68-10A6/test_drug/disp/disp_%d.png",i);
            //sprintf(buff2,"/media/lj/FA68-10A6/test_drug/disp_8/disp_%d.png",i);
            Mat src;
            //Mat dst;
            src=imread(buff1,CV_LOAD_IMAGE_UNCHANGED);
            Mat ff=Mat::zeros(src.rows,src.cols,CV_8UC1);
            for(int k=0;k<src.rows;k++){
                for(int kk=0;kk<src.cols;kk++){
                    int n=src.at<ushort>(k,kk);
                    ff.at<uchar>(k,kk)=n;
                }
            }
            //src.copyTo(dst);
            //imshow("haha",ff*100);
            //waitKey(0);
            imwrite(buff2,ff);
            cout<<i<<"  is ok"<<endl;
            file.close();
        }

    }
    cout<<number<<" images have incorrect mask json file!"<<endl;
    return 0;
}
