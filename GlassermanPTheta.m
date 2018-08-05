function [pTheta,thetaVec] = GlassermanPTheta(pncz,weights,tail)
    [N,~,NMC] = size(pncz);
    pTheta = pncz;
    thetaVec = zeros(1,NMC);
    psi = @(theta,pnc) sum(log(sum(pnc.*exp(weights.*theta),2)),1);
    for i=1:NMC
        pnc = pncz(:,:,i);

        threshold = sum(sum(weights.*pnc,2),1);
        if tail > threshold
            energy = @(theta) psi(theta,pnc) - tail*theta;
            options = optimset('LargeScale','off', 'display', 'off');
            intialGuess = 0;
            if(i > 1); intialGuess = thetaVec(i-1); end
            [theta, fval, exitflag, output] = fminunc(energy, intialGuess, options);
            twist = pnc.*exp(weights.*theta(end));
            s = sum(twist,2);
            pTheta(:,:,i) = bsxfun(@rdivide,twist,s);
            thetaVec(i) = theta;
        end
    end
end