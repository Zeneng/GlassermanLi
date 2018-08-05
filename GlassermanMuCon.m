function [ mu, obj,exit] = GlassermanMuCon(z0,theta0, H, BETA, tail, EAD, LGC, noHess,Foption)

    weights = EAD.*LGC;
    exit=[];
    v0 = [z0;theta0];
    function [p] = psi(theta,pnc,weights)
        p = sum(log(sum(pnc.*exp(weights.*theta),2)),1);
    end

    function pncz = ComputePNC(H,BETA,z)
            [N,C] = size(H);
            NZ = 1;
            denom = (1-sum(BETA.^2,2)).^(1/2);
            BZ = BETA*z;
            CH = H;
            CHZ = repmat(CH,1,1,NZ);
            BZ = reshape(BZ,N,1,NZ);
            CBZ = repelem(BZ,1,C);
            PINV = (CHZ - CBZ) ./ denom;
            PHI = normcdf(PINV);
            PHI = [zeros(N,1,NZ) PHI];
            pncz = diff(PHI,1,2); %column wise diff
    end
    
    function [gp] = ComputeGradP(H,BETA,z)
        [N,C] = size(H);
        S = size(BETA,2);
        denom = (1-sum(BETA.^2,2)).^(1/2);
        BZ = BETA*z;
        CH = H;
        CHZ = CH;
        BZ = reshape(BZ,N,1,1);
        CBZ = repelem(BZ,1,C);
        PINV = (CHZ - CBZ) ./ denom;
        PINV = [repelem(-inf,N,1,1) PINV];
        ePINV = exp(-0.5.*PINV.^(2));
        h = (1/sqrt(2*pi)).*diff(ePINV,1,2);
        gp = zeros(N,C,S);
        for n=1:N
           for k=1:C
               gp(n,k,:) = -h(n,k)*BETA(n,:) ./ denom(n);
           end 
        end
    end

    function [energy, grad] = fu(v, H, BETA, tail, weights)
        theta = v(end);
        z = v(1:end-1);
        pnc = ComputePNC(H,BETA,z);
        energy = theta*tail - psi(theta,pnc,weights) + 0.5*(z'*z);
        
        gp = ComputeGradP(H,BETA,z);
        gPsi = sum(sum(gp.*exp(weights.*theta),2) ./ sum(pnc.*exp(weights.*theta),2),1);
        gPsi = reshape(gPsi,size(z));
        
        if nargout > 1
        
        grad = [-gPsi + z; ...
            tail - sum(sum(pnc.*weights.*exp(weights.*theta),2) ./ sum(pnc.*exp(weights.*theta),2),1)
            ];
        end
    end

    function [c,ceq,gradc,gradceq] = fl(v, H, BETA, tail, weights)
        c = 0;
        
        theta = v(end);
        z = v(1:end-1);
        pnc = ComputePNC(H,BETA,z);
        ceq = -tail + sum(sum(pnc.*weights.*exp(weights.*theta),2) ./ sum(pnc.*exp(weights.*theta),2),1);
        
        if nargout > 2
        gp = ComputeGradP(H,BETA,z);
        
        gradc = zeros(size(v));
        
        gdPsidThetaL = sum(sum(gp.*weights.*exp(weights.*theta),2) ./ ...
        sum(pnc.*exp(weights.*theta),2),1);
        
        gdPsidThetaR =   sum(sum(gp.*exp(weights.*theta),2) .* ...
         sum(pnc.*weights.* exp(weights.*theta),2) ./ (sum(pnc.* exp(weights.*theta),2)).^2,1);
        
        gdPsidTheta =  gdPsidThetaL - gdPsidThetaR;
        gdPsidTheta = reshape(gdPsidTheta,size(z));
        
        ddPsiddThetaL = sum(sum(pnc.*(weights.^2).*exp(weights.*theta),2) ./ ...
                       sum(pnc.*exp(weights.*theta),2),1);
        ddPsiddThetaR = sum(((sum(pnc.*weights.*exp(weights.*theta),2)).^2) ./ ...
                            (sum(pnc.*exp(weights.*theta),2).^2),1);

        ddPsiddTheta = ddPsiddThetaL - ddPsiddThetaR;

        
        gradceq = [gdPsidTheta;ddPsiddTheta];
        end
    end

    function [hessian] = hessianPZ(z, H, BETA)
        [N,C] = size(H);
        S = size(BETA,2);
        denom = (1-sum(BETA.^2,2)).^(1/2);
        BZ = BETA*z;
        CH = H;
        CHZ = CH;
        BZ = reshape(BZ,N,1,1);
        CBZ = repelem(BZ,1,C);
        PINV = (CHZ - CBZ) ./ denom;
        PINV = [repelem(0,N,1,1) PINV];
        PINV(PINV == inf) = 0;
        ePINV = -1*PINV.*exp(-0.5.*PINV.^(2));
        h = (1/sqrt(2*pi)).*diff(ePINV,1,2);
        hessian = zeros(N,C,S,S);
        for n=1:N
           for k=1:C
               hessian(n,k,1:S,1:S) = h(n,k)*(BETA(n,:)'*BETA(n,:)) ./ (denom(n).^2);
           end 
        end
    end

    function [hessian] = hessianPsiZ(v, H, BETA, tail, weights, pnc)
        theta = v(end);
        z = v(1:end-1);
        S = size(z,1);
        gpz = ComputeGradP(H,BETA,z);
        hpz = hessianPZ(z, H, BETA);
        hLeft = sum(sum(hpz.*exp(weights.*theta),2),1) ./ sum(sum(pnc.*exp(weights.*theta),2),1);
        hLeft = reshape(hLeft,S,S,1,1);
        a = sum(sum(gpz.*exp(weights.*theta),2),1);
        a = reshape(a,S,1,1);
        b = sum(sum(pnc.*exp(weights.*theta),2),1)^.2;
        hRight = (a*a') ./ b;
        hessian = hLeft + hRight;
    end

    function [hessian] = hessianE(v, lambda, H, BETA, tail, weights)
        theta = v(end);
        z = v(1:end-1);
        S = size(z,1);
        pnc = ComputePNC(H,BETA,z);
        gp = ComputeGradP(H,BETA,z);
        hPsiZ = hessianPsiZ(v, H, BETA, tail, weights, pnc);
        hPsiZ = reshape(hPsiZ,S,S,1,1);
        
        gdPsidThetaL = sum(sum(gp.*weights.*exp(weights.*theta),2) ./ ...
        sum(pnc.*exp(weights.*theta),2),1);
        
        gdPsidThetaR =   sum(sum(gp.*exp(weights.*theta),2) .* ...
         sum(pnc.*weights.* exp(weights.*theta),2) ./ (sum(pnc.* exp(weights.*theta),2)).^2,1);
        
        gdPsidTheta =  gdPsidThetaL - gdPsidThetaR;
        gdPsidTheta = reshape(gdPsidTheta,size(z));
        
        ddPsiddThetaL = sum(sum(pnc.*(weights.^2).*exp(weights.*theta),2) ./ ...
                       sum(pnc.*exp(weights.*theta),2),1);
        ddPsiddThetaR = sum(((sum(pnc.*weights.*exp(weights.*theta),2)).^2) ./ ...
                            (sum(pnc.*exp(weights.*theta),2).^2),1);

        ddPsiddTheta = ddPsiddThetaL - ddPsiddThetaR;
        
        hessian = zeros(S+1,S+1);
        hessian(1:S,1:S) = -hPsiZ;
        hessian(S+1,1:S) = -gdPsidTheta;
        hessian(1:S,S+1) = -gdPsidTheta;
        hessian(S+1,S+1) = -ddPsiddTheta;
        
        hessian = hessian - lambda.eqnonlin*hessian; % hessian of constraint is negative of current hessian
        
        hessian(1:S,1:S) = hessian(1:S,1:S) + ones(S,S);
  
    end

%%%%%%%%%%%uncontrianted objective funciton
function [pTheta,thetaVec] = GlassermanPTheta(pncz,weights,tail)
    pTheta = pncz;
    thetaVec = zeros(1,1);
    psi = @(theta,pnc) sum(log(sum(pnc.*exp(weights.*theta),2)),1);
    for k=1:1
        pnc = pncz(:,:,k);

        threshold = sum(sum(weights.*pnc,2),1);
        if tail > threshold
            energy = @(theta) psi(theta,pnc) - tail*theta;
            option = optimset('LargeScale','off', 'display', 'off');
            intialGuess = 0;
            if(k > 1); intialGuess = thetaVec(k-1); end
            [theta,~] = fminunc(energy, intialGuess, option);
            twist = pnc.*exp(weights.*theta(end));
            s = sum(twist,2);
            pTheta(:,:,k) = bsxfun(@rdivide,twist,s);
            thetaVec(k) = theta;
        end
    end
end

function [energy, grad] = fu1(z, H, BETA, tail, weights)
        pnc = ComputePNC(H,BETA,z);
        [~,theta] = GlassermanPTheta(pnc,weights,tail); 
        energy = theta*tail - psi(theta,pnc,weights) + 0.5*(z'*z);
        gp = ComputeGradP(H,BETA,z);
        gPsi = sum(sum(gp.*exp(weights.*theta),2) ./ sum(pnc.*exp(weights.*theta),2),1);
        gPsi = reshape(gPsi,size(z));
        
        if nargout > 1
            [c,ceq,gradc,gradceq] = fl([z;theta], H, BETA, tail, weights);
            delta = gradceq(1:(end-1))./gradceq(end);
            gradceq(1:(end-1))
            gradceq(end)
            grad = -gPsi + z -(tail - sum(sum(pnc.*weights.*exp(weights.*theta),2) ./ sum(pnc.*exp(weights.*theta),2),1))*delta;
         
     

        end
    end


    Energy = @(v) fu(v, H, BETA, tail, weights);
    Energy1 = @(z) fu1(z, H, BETA, tail, weights);
    Condition = @(v) fl(v, H, BETA, tail, weights);
    HessianFcn = @(v,lambda) hessianE(v, lambda, H, BETA, tail, weights);
    method1={'interior-point'};
    method2={'quasi-newton'};
    if Foption == true
        
      for i=1:length(method1)
         if noHess == true
    
            options = optimoptions ('fmincon','Algorithm',char(method1(i)),'MaxFunctionEvaluations',13000,...
            'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,'CheckGradients',false,'StepTolerance',1e-20,'FunctionTolerance',1e-20,...
            'OptimalityTolerance',1e-15,'maxIter',3000); 
            [v,fval,exitflag,output,lambda,grad,hessian] = fmincon(Energy,v0,[],[],[],[],[],[],Condition,options);
             mu(:,i) = v(1:end-1);
             obj(i)=fval;
             exit(i)=exitflag;
            
         
         else
             options = optimoptions ('fmincon','Algorithm',char(method(i)),'MaxFunctionEvaluations',13000,...
             'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,'CheckGradients',false,'StepTolerance',1e-15,'FunctionTolerance',1e-15,...
             'maxIter',3000,'HessianFcn',HessianFcn);
             [v,fval,exitflag,output,lambda,grad,hessian] = fmincon(Energy,v0,[],[],[],[],[],[],Condition,options);
             mu(:,i) = v(1:end-1);
             obj(i)=fval;
             exit(i)=exitflag;
         end
         
         
             
             
      end 
    
    else
        for i=1:length(method2)
         if noHess == true
    
            options = optimoptions ('fminunc','Algorithm',char(method2(i)),'MaxFunctionEvaluations',13000,...
            'SpecifyObjectiveGradient',true,'CheckGradients',false,'StepTolerance',1e-100,'FunctionTolerance',1e-100,....
            'maxIter',6000,'OptimalityTolerance',1.0e-015,'PlotFcn','optimplotfval');
            [z1,fval,exitflag,~] = fminunc(Energy1,z0,options);
             mu(:,i) = z1;
             obj(i)=fval;
             exit(i)=exitflag;
            
         
         else
             options = optimoptions ('fminunc','Algorithm',char(method2(i)),'MaxFunctionEvaluations',13000,...
             'SpecifyObjectiveGradient',true,'CheckGradients',false,'StepTolerance',1e-15,'FunctionTolerance',1e-15,...
             'maxIter',3000,'HessianFcn',HessianFcn);
             [z1,fval,exitflag,output,lambda,grad,hessian] = fminunc(Energy1,z0,options);
             mu(:,i) = z1;
             obj(i)=fval;
             exit(i)=exitflag;
         end
        
        
       end
    end
    

end
