#include <iostream>
#include <string>
using namespace std;

void convert(){
    string classes[7] = {"lq.csv", "tth.csv", "ttw.csv", "ttz.csv", "ttbar.csv", "vv.csv", "other.csv"};
    string roots[7] = {"lq.root", "tth.root", "ttw.root", "ttz.root", "ttbar.root", "vv.root", "other.root"};
  //  for(int i = 0; i < 7; ++i){
        //string PATH = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/lq_all/root/";
        string PATH = "/home/lucas/Documents/KYR/bc_thesis/data_processed/final_data_analysis_weights/lq_all/lq_800/root/";
        //string clas = PATH + classes[i];
        string clas = PATH + "limit.csv";
        cout << clas << endl;
        auto rdf = ROOT::RDF::MakeCsvDataFrame(clas);
        //string rot = PATH + roots[i];
        string rot = PATH + "limit.root";
        rdf.Snapshot("nominal", rot);
  //  }
}
