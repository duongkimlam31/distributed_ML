#struct for data structure
struct communication_data{
    1: string fname,
    2: list<list<double>> _V,
    3: list<list<double>> _W,
    4: i32 epochs,
    5: double eta,
    6: bool delay
}

service coordinator{
    double train(1: string dir, 2: i32 rounds, 3: i32 epochs,  4: i32 h, 5: i32 k, 6: double eta),
    communication_data pull_data(1: double load_probability),
    void push_data(1: list<list<double>> gradient_V, 2: list<list<double>> gradient_W),
    void contact()
}

service compute_node{
    void wait_coordinator()
}

// thrift -r --gen py distributed_ml.thrift 