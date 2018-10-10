function [p,mu,sigma,bond,prob,upper,lower] = DensityAtZ(z,H,BETA,tail,EAD,LGC,NZ) 
    p = creditEvent(BETA,z,H,NZ);
    mu = computeMu(LGC,p,EAD);
    sigma = computeSigma(LGC,p,EAD);
    bond=innerbond(tail,EAD,LGC,p,mu,sigma);
    prob = 1-normcdf((tail - mu) / sigma);
    prob = max(prob, 1e-100);
    upper=prob+bond;
    upper=reshape(upper,1,NZ);
    lower=prob-bond;
    lower=reshape(lower,1,NZ);
    
   
    function [p] = creditEvent(BETA,z,H,NZ)
        N = size(BETA,1);
        C = size(H,2);
        denom = (1-sum(BETA.^2,2)).^(1/2); 
        BZ = BETA*z;
        CH = H;
        CHZ = repmat(CH,1,1,NZ);
        BZ = reshape(BZ,N,1,NZ);
        CBZ = repelem(BZ,1,C);
        PINV = (CHZ - CBZ) ./ denom;
        PHI = normcdf(PINV);
        PHI = [zeros(N,1,NZ) PHI];
        p = diff(PHI,1,2); %column wise diff
    end

    function [mu] = computeMu(LGC,p,EAD)
        mu = sum(EAD.*sum(LGC.*p,2));
    end
    
    

    function [sigma] = computeSigma(LGC,p,EAD)
        [N,C] = size(LGC);
        A = zeros(N,(C-1)*C/2);
        index = 1;
        for a=1:C
            for b=1:(a-1)
                A(:,index) = ((LGC(:,a) - LGC(:,b)).^2).*p(:,a).*p(:,b);
                index = index + 1;
            end
        end
        B = sum(A,2);
        sigma = sqrt(sum((EAD.^2).*B));
    end

    function [bond]=innerbond(tail,EAD,LGC,p,mu,sigma)
        x=(tail-mu)./sigma;
        w=EAD./sum(EAD);
        x=1+abs(x).^3;
        M=w.*sum(LGC.*p,2);
        variable=(abs((w.*LGC-M)/sigma)).^3;
        pho=sum(sum(variable.*p,2));
        c0=0.7164;
        c1= 31.395;
        bond=min(c0*pho,c1*pho./x);
    end
        
end