function Energy = Object(z, H, BETA, tail, EAD, LGC)
    weights = EAD.*LGC;
    
    
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

        pncz = ComputePNC(H,BETA,z);
        [~,thet] = GlassermanPTheta(pncz,weights,tail); 
        Energy = thet*tail - psi(thet,pncz,weights) + 0.5*(z'*z);
        
        

       
    end
