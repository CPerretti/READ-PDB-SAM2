template <class Type>
Type nllS(confSet &conf, paraSet<Type> &par, array<Type> &logS){
  using CppAD::abs;
  Type nll=0;
  int stateDimS=logS.dim[0]; // # n ages
  int timeSteps=logS.dim[1]; // # n time steps
  int stateDimN=conf.keyLogScale.dim[1];
  vector<Type> sdLogSsta=exp(par.logSdLogSsta);
  array<Type> resS(stateDimS, timeSteps-1); 
  matrix<Type> svar(stateDimS, stateDimS);  
  matrix<Type> scor(stateDimS,stateDimS);
  vector<Type> ssd(stateDimS); 
  
  if(conf.corFlagS==0){
    scor.setZero();
  }
  
  for(int i=0; i<stateDimS; ++i){
    scor(i,i) = 1.0;
  }
  
  if(conf.corFlagS==1){
    for(int i=0; i<stateDimS; ++i){
      for(int j=0; j<i; ++j){
        scor(i,j) = trans(par.itrans_rhoS(0));
        scor(j,i) = scor(i,j);
      }
    }
  }
  
  if(conf.corFlagS==2){
    for(int i=0; i<stateDimS; ++i){
      for(int j=0; j<i; ++j){
        scor(i,j)=pow(trans(par.itrans_rhoS(0)),abs(Type(i-j)));
        scor(j,i)=scor(i,j);
      }
    } 
  }
  
  int i,j;
  for(i=0; i<stateDimS; ++i){
    for(j=0; j<stateDimN; ++j){
      if(conf.keyLogScale(0,j)==i)break;
    }
    ssd(i)=sdLogSsta(conf.keyVarS(0,j));
  }
  
  for(i=0; i<stateDimS; ++i){
    for(j=0; j<stateDimS; ++j){
      svar(i,j)=ssd(i) * ssd(j) * scor(i,j);
    }
  }
  
  //density::MVNORM_t<Type> neg_log_densityS(svar);
  MVMIX_t<Type> neg_log_densityS(svar,Type(conf.fracMixS));
  Eigen::LLT< Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> > lltCovS(svar);
  matrix<Type> LS = lltCovS.matrixL();
  matrix<Type> LinvS = LS.inverse();
  
  for(int i = 1; i < timeSteps; ++i){
    resS.col(i-1) = LinvS*(vector<Type>(logS.col(i)-logS.col(i-1)));    
    nll += neg_log_densityS(logS.col(i)-logS.col(i-1));
    
  }
  
  return nll;
}
