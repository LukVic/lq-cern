
void read_events() {

    string PATH = "/home/lucas/Documents/KYR/bc_thesis/data_new_07/mc16a(1)/new7post/nominal/mc16a/p4416/312222_AF_a.root";
    
    TFile *f=new TFile("/home/lucas/Documents/KYR/bc_thesis/data_new_07/mc16a(1)/new7post/nominal/mc16a/p4416/312222_AF_a.root", "read");
    TTree *T= (TTree*)f->Get("nominal");
    int entries = T->GetEntries();

    cout << entries << endl;
    T->Show(10);

    f->Close();
}