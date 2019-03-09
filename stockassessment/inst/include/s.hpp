template <class Type>
Type nllS(confSet &conf, paraSet<Type> &par, array<Type> &logScale){ 
  Type nll=0;
  int stateDimS=logScale.dim[0]; // # n ages
  int timeSteps=logScale.dim[1]; // # n time steps
  //array<Type> resN(stateDimS,timeSteps-1); 
  matrix<Type> nvar(stateDimS,stateDimS);
  vector<Type> varLogScale=exp(par.logSdLogScale*Type(2.0));
  for(int i=0; i<stateDimS; ++i){
    for(int j=0; j<stateDimS; ++j){
      if(i!=j){nvar(i,j)=0.0;}else{nvar(i,j)=varLogScale(conf.keyVarLogScale(i));} ///<<< I THINK THIS IS THE PROBLEM AREA
                                            //varLogN(conf.keyVarLogN(i));}
    }
  }
  MVMIX_t<Type> neg_log_densityS(nvar,Type(0));
  //Eigen::LLT< Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> > lltCovS(nvar);
  //matrix<Type> LS = lltCovS.matrixL();
  //matrix<Type> LinvS = LS.inverse();

  for(int i = 1; i < timeSteps; ++i){ 
    vector<Type> predS = logScale.col(i-1);//predNFun(dat,conf,par,logN,logF,i); 
    //resS.col(i-1) = LinvS*(vector<Type>(logScale.col(i)-predS));    
    nll += neg_log_densityS(logScale.col(i)-predS); // N-Process likelihood 
    // SIMULATE_F(of){
    //   if(conf.simFlag==0){
    //     logN.col(i) = predN + neg_log_densityN.simulate();
    //   }
    // }
  }
  // if(conf.resFlag==1){
  //   ADREPORT_F(resN,of);
  // }
  // if(CppAD::Variable(keep.sum())){ // add wide prior for first state, but _only_ when computing ooa residuals
  //   Type huge = 10;
  //   for (int i = 0; i < stateDimS; i++) nll -= dnorm(logN(i, 0), Type(0), huge, true);  
  // } 
  return nll;
}
