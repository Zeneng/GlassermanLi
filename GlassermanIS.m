
N = 10000; % Number of creditors
NZ = 10; % Number of Z samples 
nE = 10; % Number of epsilion samples to take PER z sample
Nrun=1000; %Number of run times
S = 1; % Dimension of Z

%Initialize data

[H, BETA, EAD, CN, LGC, CMM, C] = ProblemParams(N, S, false);

tail=0.5;

%probability of loss bigger than tail using GlassermanLi
a1 = zeros(1,Nrun);
v1 = zeros(1,Nrun);
L1=[];
%probability of loss bigger than tail using Naive Bilevel
a2 = zeros(1,Nrun);
v2 = zeros(1,Nrun);
L2=[];
%probability of loss bigger than tail using Naive Bilevel
a3 = zeros(1,Nrun);
v3 = zeros(1,Nrun);
L3=[];

%CLT find upperbond and lowerbond
Upper=zeros(1,Nrun);
Lower=zeros(1,Nrun);
CLU=[];
CLL=[];







%Plot object function
%[X,Y] = meshgrid(-20:1:20);
%L=length(X);
%Z=zeros(L,L);
%for i=1:L
    %for j=1:L
        %Z(i,j)=Object([X(i,j);Y(i,j)],H, BETA, tail, EAD, LGC);
    %end
%end

%surf(X,Y,-Z)
%title('tail=2.0')

%one creditor analytical solution
%Anly1=normcdf(H(1));
%two creditor analytical solution
%pho=BETA(1)*BETA(2);
%Anly2=mvncdf([-Inf -Inf],[H(1) ,H(2)],[0,0],[1 pho;pho 1]);

%modify the problem into a simple constant set up
%EAD=0.2*ones(N,1);
%LGC=[ones(1,N);zeros(1,N)]';
%BETA=0.5*ones(N,1);
%pho=BETA(1)^2;

%multidimensional N must be smaller than 25
%g represents how many creditor defaults
%g=2;
%xl=[-Inf(1,g),H(1)*ones(1,N-g)];
%xu=[H(1)*ones(1,g),Inf(1,N-g)];
%mureal=zeros(0,1,N);
%Varmat=repmat(pho,N)-diag(repmat(pho-1,1,N));
%Anlym1=mvncdf(xl,xu,mureal,Varmat);
%Anlym1=nchoosek(N,g)*Anlym1;



totalT = cputime;
disp(strcat('RUN NUMBER',num2str(1)))
disp('BEGIN FINDING SHIFTED MEAN')
t = cputime;
Initial=[-1 + 2*rand(S,20),zeros(S,1)];
[~,col]=size(Initial);
Mu1=zeros(S,col);
%Mu2=zeros(S,col);
Obj1=zeros(1,S);
%Obj2=zeros(1,S);
flag=zeros(1,S);
for i=1:col
    [mu1,obj1,exit1] = GlassermanMuCon(Initial(:,i),0, H, BETA, tail, EAD, LGC, true, false);
    %[mu2,obj2,exit2] = GlassermanMuCon(Initial(:,i),0, H, BETA, tail, EAD, LGC, true, true);
    Mu1(:,i)=mu1;
    Obj1(i)=obj1;
    flag(i)=exit1;
    %Mu2(:,i)=mu2;
    %Obj2(i)=obj2;
end
    

[~,idx1] = min(Obj1);
%[~,idx2] = min(Obj2);
%if min(obj1)<min(obj2)
    %mu=Mu1(:,idx1);
%else
    %mu=Mu2(:,idx2);
%end
mu=Mu1(:,idx1);
%pattern search
%fun=@(z)Object(z,H,BETA,tail,EAD,LGC);
%patternsearch(fun,Mu1(:,1));
    
disp(strcat('FINISH FINDING SHIFTED MEAN...',num2str(cputime - t),'s'))

for r=1:Nrun 
    
    disp('BEGIN SAMPLING')
    t = cputime;
    sampleZ = mvnrnd(mu,eye(S),NZ)';
    disp(strcat('FINISH SAMPLING...',num2str(cputime - t),'s'))

    disp('BEGIN COMPUTING PNCZ')
    t = cputime;
    denom = (1-sum(BETA.^2,2)).^(1/2);
    BZ = BETA*sampleZ;
    CH = H;
    CHZ = repmat(CH,1,1,NZ);
    BZ = reshape(BZ,N,1,NZ);
    CBZ = repelem(BZ,1,C);
    PINV = (CHZ - CBZ) ./ denom;
    PHI = normcdf(PINV);
    PHI = [zeros(N,1,NZ) PHI];
    pncz = diff(PHI,1,2); %column wise diff
    disp(strcat('FINISH COMPUTING PNCZ...',num2str(cputime - t),'s'))
    
    disp('BEGIN COMPUTING THETA')
    t = cputime;
    weights = EAD.*LGC;
    [pTheta,theta] = GlassermanPTheta(pncz,weights,tail);
    disp(strcat('FINISH COMPUTING THETA...',num2str(cputime - t),'s'))

    disp('BEGIN SAMPLING PNCZ')
    t = cputime;
    cdf = cumsum(pTheta,2);
    cdf = repelem(cdf,1,1,nE);
    u = rand([N,1,nE*NZ]);
    isOne = (cdf >= u) == 1;
    ind = (cumsum(isOne,2) == 1);
    disp(strcat('FINISH SAMPLING PNCZ...',num2str(cputime - t),'s'))

    disp('BEGIN COMPUTING LOSS')
    t = cputime;
    LossMat = repelem(weights,1,1,NZ*nE).*ind;
    Loss = sum(sum(LossMat,2),1);
    theta = reshape(theta,[1,1,NZ]);
    B = zeros([N C NZ]);
    for j=1:NZ
        B(:,:,j) = theta(:,:,j)*weights;
    end
    psi = sum(log(sum(pncz.*exp(B),2)),1);
    LRE = reshape(exp(-repelem(theta,1,1,nE).*Loss + repelem(psi,1,1,nE)),1,nE*NZ,1);
    LRZ = repelem(arrayfun(@(i) exp(-mu'*sampleZ(:,i) + 0.5*(mu'*mu)),1:NZ),1,nE);
    LR = LRE.*LRZ;
    Loss = reshape(Loss,1,nE*NZ);
    l1 = double(Loss >= tail).*LR;
    L1=[L1 l1];
    a1(r) = mean(vpa(L1));
    v1(r) = var(vpa(L1));
   
    disp(strcat('FINISH COMPUTING LOSS...',num2str(cputime - t),'s'))
    disp(strcat('TOTAL RUNTIME...',num2str(cputime - totalT),'s'))
    
    
    %%%%%%%%%%%%%%%%%naive way
    disp('BEGIN Naive SAMPLING')
    t = cputime;
    sampleZ = randn(S,NZ);
    sampleE = randn(N,nE*NZ);
    disp(strcat('FINISH SAMPLING...',num2str(cputime - t),'s'))
    
    %calulate upper and lower bond
    [~,~,~,~,~,upperb,lowerb]=DensityAtZ(sampleZ,H,BETA,tail,EAD,LGC,NZ);
    CLU=[CLU,upperb];
    CLL=[CLL,lowerb];
    len=length(CLU);
    t=tinv(0.95,len);
    Upper(r) = mean(CLU)+t*sqrt(var(CLU)/len);
    Lower(r) = mean(CLL)-t*sqrt(var(CLL)/len);
    

    disp('BEGIN COMPUTING Y')
    t = cputime;
    denom = (1-sum(BETA.^2,2)).^(1/2); %Not used as denom but keeping notation consistant
    BZ = BETA*sampleZ;
    BZ = reshape(BZ,N,1,NZ);
    BZ = repelem(BZ,1,1,nE);
    sampleE = reshape(sampleE,N,1,nE*NZ);
    Y = BZ + bsxfun(@times,sampleE,denom);
    disp(strcat('FINISH COMPUTING Y...',num2str(cputime - t),'s'))
    
    disp('BEGIN COMPUTING INDICATORS')
    t = cputime;
    CH = H;
    CHZE = repmat(CH,1,1,nE*NZ);
    isOne = ((Y <= CHZE) == 1);
    ind = isOne & (cumsum(isOne,2) == 1);
    disp(strcat('FINISH COMPUTING INDICATORS...',num2str(cputime - t),'s'))

    disp('BEGIN COMPUTING LOSS')
    t = cputime;
    weights = EAD.*LGC;
    LossMat = repelem(weights,1,1,NZ*nE).*ind;
    Loss = sum(sum(LossMat,2),1);
    Loss = reshape(Loss,1,nE*NZ);
    l2 = double(Loss >= tail);
    L2=[L2 l2];
    a2(r) = vpa(mean(L2));
    v2(r) = vpa(var(L2));
    disp(strcat('TOTAL RUNTIME...',num2str(cputime - totalT),'s'))
    
    %%%%%%%%%%%%%%%%%%another MC
    disp('BEGIN SAMPLING')
    t = cputime;
    sampleZ = randn(S,NZ);
    sampleE = randn(N,nE*NZ);
    disp(strcat('FINISH SAMPLING...',num2str(cputime - t),'s'))

    disp('BEGIN COMPUTING Y')
    t = cputime;
    denom = (1-sum(BETA.^2,2)).^(1/2); %Not used as denom but keeping notation consistant
    BZ = BETA*sampleZ;
    BZ = reshape(BZ,N,1,NZ);
    BZ = repelem(BZ,1,1,nE);
    sampleE = reshape(sampleE,N,1,nE*NZ);
    Y = BZ + bsxfun(@times,sampleE,denom);
    disp(strcat('FINISH COMPUTING Y...',num2str(cputime - t),'s'))
    
    disp('BEGIN COMPUTING INDICATORS')
    t = cputime;
    CH = H;
    CHZE = repmat(CH,1,1,nE*NZ);
    isOne = ((Y <= CHZE) == 1);
    ind = isOne & (cumsum(isOne,2) == 1);
    disp(strcat('FINISH COMPUTING INDICATORS...',num2str(cputime - t),'s'))

    disp('BEGIN COMPUTING LOSS')
    t = cputime;
    weights = EAD.*LGC;
    LossMat = repelem(weights,1,1,NZ*nE).*ind;
    Loss = sum(sum(LossMat,2),1);
    Loss = reshape(Loss,1,nE*NZ);
    l3 = double(Loss >= tail);
    L3=[L3 l3];
    a3(r) = vpa(mean(L3));
    v3(r) = vpa(var(L3));
    disp(strcat('TOTAL RUNTIME...',num2str(cputime - totalT),'s'))
    
    
    
end

figure(2)
plot(NZ*nE:NZ*nE:NZ*nE*Nrun,a1,NZ*nE:NZ*nE:NZ*nE*Nrun,Upper,NZ*nE:NZ*nE:NZ*nE*Nrun,Lower,NZ*nE:NZ*nE:NZ*nE*Nrun,a2,NZ*nE:NZ*nE:NZ*nE*Nrun,a3)
legend('GlassermanLi','Upper','Lower','Naive1','Naive2')
title('Mean')
xlabel('Run number')

figure(3)
plot(NZ*nE:NZ*nE:NZ*nE*Nrun,v1,NZ*nE:NZ*nE:NZ*nE*Nrun,v2,NZ*nE:NZ*nE:NZ*nE*Nrun,v3)
legend('GlassermanLi','Naive1','Naive2')
title('Variance')
xlabel('Run number')

disp('GlassermanLi mean')
vpa(a2(end))
disp('GlassermanLi var')
vpa(v1(end))
disp('naive mean')
vpa(a2(end))
disp('naive var')
vpa(v2(end))


    

