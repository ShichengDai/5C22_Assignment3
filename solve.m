function [F, B, alpha] = solve(mean_F, Sigma_F, mean_B, Sigma_B, C, sigma_C, alpha_init, maxIter, minLike)

    I = eye(3);
    maxLike = -realmax;
    
    for i = 1:size(mean_F, 2)
        mean_Fi = mean_F(:, i);
        invSigma_Fi = inv(Sigma_F(:, :, i));
                
        for j = 1:size(mean_B, 2)
            mubi = mean_B(:, j);
            invSigmabi = inv(Sigma_B(:, :, j));
            
            alpha = alpha_init;
            iter = 1;
            lastLike = -realmax;
            
            while (1)
                
                    % solve for F,B
					alpha_sq_inv = alpha^2/sigma_C^2;
					one_alpha_sq_inv = (1-alpha)^2/sigma_C^2;
					alpha_one_alpha_inv = alpha*(1-alpha)/sigma_C^2;
					A = [invSigma_Fi+I*alpha_sq_inv, I*alpha_one_alpha_inv;                  I*alpha_one_alpha_inv, invSigmabi+I*one_alpha_sq_inv];
					b = [invSigma_Fi*mean_Fi + C*(alpha/sigma_C^2);                  invSigmabi*mubi + C*((1-alpha)/sigma_C^2)];
					X = A\b;
					F = max(0,min(1,X(1:3)));
					B = max(0,min(1,X(4:6)));
					
					% solve for alpha
					F_minus_B = F - B;
					alpha_num = (C-B)'*F_minus_B;
					alpha_den = sum(F_minus_B.^2);
					alpha = max(0,min(1,alpha_num/alpha_den));
					
					% calculate likelihood
					C_minus_alphaF_minus_1minusalphaB = C - alpha*F - (1-alpha)*B;
					L_C = -sum(C_minus_alphaF_minus_1minusalphaB.^2)/sigma_C;
					L_F = -(F-mean_Fi)'*invSigma_Fi*(F-mean_Fi)/2;
					L_B = -(B-mubi)'*invSigmabi*(B-mubi)/2;
					like = L_C + L_F + L_B;
					
					if iter >= maxIter || abs(like-lastLike) <= minLike
						break;
					end
					
					lastLike = like;
					iter = iter + 1;
            end
            
            % keep only the maximum likelihood estimates
            if like > maxLike
                maxLike = like;
                maxF = F;
                maxB = B;
                maxAlpha = alpha;
            end
        end
    end
    
    % return maximum likelihood estimates
    F = maxF;
    B = maxB;
    alpha = maxAlpha;
    
end
