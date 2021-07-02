% loadData
clear; close all
filenames = {'1-3001-3300frame','4-3065-3365frame'} ;
mat_dir = '.\'; % data_mat\
video_dir = '.\videos\';
dim_xy = 3 ;

addpath('.\heatmaps')

% analyzed data
% mat_dir0 = '..\..\weights\bats_gvar_2\' ; % for other than fujii
mat_dir0 = '\\spica\workspace4\fujii\work\ABM\weights\bats_gvar_2\' ; % for fujii

% mat_dir1 = [mat_dir0,'_TEST_bidirection\'];
mat_dir2 = [mat_dir0,'_TEST_percept_CF_pred_self\'];

% movie
load([mat_dir,'dataset_bats']) ;

if 1
    order = 3 ;
    for f = 1:length(filenames)

        if f==1
            List = {'1','2','3','4','5','6','7'};
        else
            List = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',...
                '21','22','23','24','25','26','27'};
        end
        
        % GVAR
%         load([mat_dir1,'coeffs_',num2str(f)]) ;
%         coeffs_gvar = squeeze(coeffs) ; 
%         coeffs_time_gvar = squeeze(coeffs_time)/max(abs(coeffs_time(:))) ;
%         coeffs_gvar_ = coeffs_gvar; 
%         coeffs_gvar_(coeffs_gvar_>0) = 1 ;
%         coeffs_gvar_(coeffs_gvar_<0) = -1 ;

        % Our method
        load([mat_dir2,'coeffs_',num2str(f)]) ;
        
        n = 1 ;
        pos = dataset{f}.loc_nan ;
        label = dataset{f}.label ;
        data_nan = squeeze(dataset{f}.data_nan) ;
        Start = 1 ; End = size(pos,1)-order-1 ;
        Fs = dataset{f}.Fs ;
        Time = 1/Fs:1/Fs:End/Fs ;
        K = size(pos,3) ;
        max_xy = dataset{f}.max_xy ;
        min_xy = dataset{f}.min_xy ;
        coeffs_ = coeffs_time ;
        coeffs = squeeze(coeffs) ; 
        
        % binary
        max_coeffs_ = max(coeffs_(:)) ;
        min_coeffs_ = min(coeffs_(:)) ;
        coeffs_binary = zeros(size(coeffs_,1),K,K);

        
        % values
        for k = 1:K
            jj = 1 ; % jjj = 1 ;
            
            for j = 1:K
                if j ~= k
                    coeffs_(data_nan(:,j)==1,k,jj) = NaN ;
                    coeffs_(coeffs_(:,k,jj)==0,k,jj) = NaN ;
                    coeffs_binary(coeffs_(:,k,jj)>=max_coeffs_/2,k,jj) = 1 ;
                    coeffs_binary(coeffs_(:,k,jj)<=min_coeffs_/2,k,jj) = -1 ;
                    coeffs_binary(isnan(coeffs_(:,k,jj)),k,jj) = NaN ;
                    %if sum(isnan(coeffs_(:,k,jj))) < End-order
                    %    plot(squeeze(coeffs_(:,k,jj))); hold on
                    %end
                    jj = jj + 1 ;
                else 
                    coeffs_binary(:,k,jj) = NaN ;
                end
            end
        end
        y_max = median(nanmax(nanmax(abs(coeffs_),[],1),[],2)) ;
        
        dataK = reshape(data(1,args.K:args.K+End-1,:),End,args.num_dims,K);
        % weight 
        weight1 = weights(1);%(:,:,1);

        % overview  
        if 1
            figure(1000+f)
            coeffs_raw_ = zeros(K,K) ;
            
            for k = 1:K
                jj = 1 ; 
                for j = 1:K
                    if k ~= j
                        if sum((data_nan(:,k)-1).*(data_nan(:,jj)-1)) > 0
                            coeffs_raw_(k,j) = coeffs(k,jj) ; % coeffs_raw
                        else coeffs_raw_(k,j) = NaN ;
                        end
                        jj = jj + 1 ;
                    else coeffs_raw_(k,j) = NaN ;
                    end
                end
            end
            %subplot 121
            heatmap(coeffs_raw_, List, List, [], 'Colormap', 'jet', 'NaNColor', [1 1 1],'Colorbar', true,...
                'MinColorValue',-1,'MaxColorValue',1);%*100'%0.2f', 'ColorLevels', 5
            title(['bats ',num2str(f), ' Granger causality matrix'])
            % xlabel('cause ID'); ylabel('effect ID')
        
        % count 
        for l = 1:2 % label
            n_l = sum(label==l) ;
            count_coeffs(l,1,f) = nansum(nansum(triu(coeffs_raw_(label==l,label==l))==1)) ;
            count_coeffs(l,2,f) = nansum(nansum(triu(coeffs_raw_(label==l,label==l))==-1)) ;
            count_coeffs(l,3,f) = nansum(nansum(triu(coeffs_raw_(label==l,label==l))==0)) - (n_l*(n_l-1)/2) ;
            count_coeffs(l,4,f) = sum(count_coeffs(l,1:3,f)) ;
            count_coeffs(l,5,f) = sum(label==l) ;
        end
        end
        % percept
        dist = dataK(:,end-K+2:end,:) ;
        
        % figure (time series)
        if 0%f == 1 
        figure(f)
        for k = 1:K
            % legend
            List_ = List;
            jj = K-1 ;
            for j = K:-1:1
                if j == k
                    List_(j) = [];
                else
                    jj = jj - 1 ;
                end
            end
            for j = 1:K-1
                % subplot(K,K-1,(k-1)*(K-1)+j)
                jj = str2num(List_{j}) ;
                subplot(K,K,(k-1)*K+jj)
                plot([0 End],[0 0],'k-'); hold on %
                plot(squeeze(coeffs_(1:End,k,j)/y_max));hold on
                
                % plot(squeeze(true_time(1:End,k,j)));
                % plot(squeeze(true_signed(1:End,k,j)));hold on
                % plot(squeeze(coeffs_(1:End,k,j))/y_max); 
                
                xlim([0 End]); ylim([-1 1])
                ylabel([num2str(k),'<-',List_{j}])
                title(['ours: ',num2str(coeffs_(k,j))]); %,' gvar: ',num2str(coeffs_gvar_(k,j))])
                hold off
            end
        end        
        end
        
        if 0%f==1 % movie 
        if f > 1
            close(h)
        end
        % figure('visible','off'); % h = figure(1); h =
        h = figure(10);
        set(gcf,'color',[1 1 1]) ;
        videoPath = [video_dir,filenames{f}]; % ['video_',num2str(n)] ;
        
        v = VideoWriter([videoPath,'_analyzed.mp4'],'MPEG-4');
        open(v)
        
        nn = 1;
        clear mov
        duration = 30 ;
        
        for t = Start:End-order
            % k = 2 ;
            jjj = 1 ;
            % legend
            
            jj = K-1 ;
            
            
            if 1 % Granger causality matrix
                if t ~= Start
                    cla(h3); 
                end
                h3 = subplot(4, 2, [4 6]) ;
                heatmap(squeeze(coeffs_binary(t,:,:)), List, List, [], 'Colormap', 'jet', 'NaNColor', [1 1 1],...
                    'Colorbar', true,'MinColorValue',-1,'MaxColorValue',1);axis equal
                title(['Bats file',num2str(f), ' Granger causality matrix'])
                
            else % time series
                for k = 1:3
                    List_ = List;
                    for j = K:-1:1
                        if j == k
                            List_(j) = [];
                        else
                            jj = jj - 1 ;
                        end
                    end
                    for j = 1:2 % K-1
                        subplot(K-1,2, jjj*2)
                        plot([0 End],[0 0],'k-'); hold on %
                        plot(squeeze(coeffs_(1:End,k,j)/y_max));hold on
                        plot(squeeze(coeffs_time_gvar(1:End,k,j)),'k:'); hold on %
                        % plot(squeeze(true_time(1:End,k,j)));hold on
                        
                        % plot(squeeze(coeffs_(1:End,k,j))/y_max);
                        plot([t-order t-order],[-1 1],'k-')
                        xlim([0 End]); ylim([-1 1])
                        ylabel([num2str(k),'<-',List_{j}])
                        hold off
                        jjj = jjj + 1 ;
                    end
                end
                % motion
                subplot(6,2,9)
                plot(0,0); hold on 
                plot(0,0,'k:')
                axis off
                legend('ours','GVAR','Location','southeast')
                hold off
                if t ~= Start
                    cla(h2); 
                end
                h2 = subplot(6,2,11) ;
                text(0,1,'-> positive: attraction'); 
                text(0,0,'-> negative: repulsion');
                xlim([-1,1]); ylim([0,1]);
                axis off
            end

            
            
            

            subplot(1, 2, 1) 
            % subplot(3,2,[1,3])
            kk = 1 ;
            for k = 1:dataset{f}.K_max 
                if ~isempty(find(k==dataset{f}.rawNo,1))% dataset{f}.followed(k) == 1
                    marker = 'o';
                    xy = pos(t,:,kk) ;
                    if t <= duration
                        xy_long = pos(1:t,:,kk) ;
                    else; xy_long = pos(t-duration:t,:,kk) ;
                    end
                    ms = 12 ; lw = 1 ;
                    if mod(k,5) == 0 ; clr = 'r' ;
                    elseif mod(k,5) == 1; clr = 'g' ;
                    elseif mod(k,5) == 2; clr = 'b' ; %
                    elseif mod(k,5) == 3; clr = 'k' ;
                    elseif mod(k,5) == 4; clr = 'm' ;
                    end

                    % plot3(xy(1),xy(2),xy(3),marker,'markersize',ms,'linewidth',lw,'color',clr); 
                    plot3(xy_long(:,1),xy_long(:,2),xy_long(:,3),'-','color',clr); hold on;hold on;
                    %text(xy(1),xy(2),xy(3),num2str(k));
                    text(xy(1),xy(2),xy(3),num2str(kk));
                    
                    kk = kk + 1 ;
                end
            end
            hold off
            
            axis equal
            set(gca,'xlim',[min_xy(f,1) max_xy(f,1)],'ylim',[min_xy(f,2) max_xy(f,2)],...
                'zlim',[min_xy(f,3) max_xy(f,3)],'View', [-10 14]) ; % [-84,12]);%
            xlabel('x')
            ylabel('y')
            zlabel('z')
            %axis off
            box off
            
            title([filenames{f},sprintf(', %d bats, Frame %d (%dHz)', K, t, floor(Fs))],'Fontsize',8) ; % Video %04d n,
            
            mov(nn)= getframe(gcf);
            drawnow
            
            nn = nn+1;
            
        end
        writeVideo(v,mov)
        close(v)
        end
    end
end

if 1% chi2 analysis for attraction-repulsion imbalance
count_all = sum(count_coeffs,3) ;
count_all(3,1:5) = sum(count_all(1:2,1:5),1) ;
count_all(4:5,1:4) = count_all(1:2,1:4)./repmat(count_all(1:2,4),1,4) ;

chi2 = chi2test(count_all(1:2,1:2),1) ;

% fishertest
% x = table(count_all(1,1:2)',count_all(2,1:2)','VariableNames',{'attraction','repulsion'},'RowNames',{'go-out','go-in'}) ;
% h = fishertest(x)
end
