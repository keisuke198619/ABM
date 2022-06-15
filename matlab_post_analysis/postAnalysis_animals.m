% loadData
clear; close all
filenames = {'mice','mice'}; % {'sula03','sula05'} ;%,'peregrine',
addpath('.\heatmaps')
datatype = 1 ;
% mat_dir = '.\data_mat\';
if contains(filenames{1},'sula')
    filename = 'sula' ;
else
    filename = filenames{datatype} ;
end
video_dir = '.\videos\';
% mat_dir = ['..\weights\',filename,'_gvar_5\_TEST_percept_CF_pred_self\']; 
T_sula = [20000,200] ;

% analyzed data
List = {'1','2','3'};
% movie

if 1
    for f = 1:length(filenames)
        if contains(filenames{f},'peregrine')
            dim_xy = 3 ;
        else
            dim_xy = 2 ;
        end
        n = 1 ;
        load([mat_dir,'coeffs_',num2str(f)]) ;
        % coeffs = coeffs/max(coeffs(:)) ;
        order = args.K ;
        
        if contains(filenames{f},'sula')
            Start = 1 ; 
            End =  200;
        else 
            Start = 1 ; 
            End = size(data,2)-order ;
        end
        %data_nan = squeeze(dataset{f}.data_nan) ;
        K = size(coeffs_time,2) ; 
        if K == 2 
            coeffs_(:,:,2) = coeffs_time ;
            coeffs_(:,:,1) = coeffs_time ;
        else
            coeffs_ = coeffs_time ;
        end
        
        
        % values
        y_max = median(nanmax(nanmax(abs(coeffs_),[],1),[],2)) ;
        % vel,loc,range,v_dir,dist
        dataK = reshape(data(1,args.K:args.K+End-1,:),End,args.num_dims,K);
        pos = dataK(:,dim_xy+1:dim_xy*2,:) ;
        
        max_xy = max(max(pos,[],1),[],3);
        min_xy = min(min(pos,[],1),[],3);
        % weight 
        weight1 = weights(1); 
        
        % percept
        dist = dataK(:,end-K+2:end,:) ;      
        
        Fs = args.Fs ;
        Time = 1/Fs:1/Fs:End/Fs ;

       
        % figure
        if 0 figure(f)
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
                
                plot(squeeze(coeffs_(1:End,k,j)/y_max));hold on
                % plot(squeeze(true_time(1:End,k,j)));
                % plot(squeeze(true_signed(1:End,k,j)));hold on

                xlim([0 End]); ylim([-1 1])
                ylabel([num2str(k),'<-',List_{j}])
                hold off
            end
        end
        end
        
        if 1 % movie
            if f > 1
                close(h)
            end
            % figure('visible','off'); % h = figure(1); h =
            h = figure(10);
            set(gcf,'color',[1 1 1]) ;
            if contains(filenames{f},'sula')
                videoPath = [video_dir,filenames{f},'_T_',num2str(Start+T_sula(f)),'_',num2str(End+T_sula(f))]; 
            else
                videoPath = [video_dir,filenames{f},num2str(f),'_T_',num2str(Start),'_',num2str(End)]; % ['video_',num2str(n)] ;
            end
            
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
                
                for k = 1:K
                    List_ = List;
                    for j = K:-1:1
                        if j == k
                            List_(j) = [];
                        else
                            jj = jj - 1 ;
                        end
                    end
                    for j = 1:K-1
                        subplot(K*(K-1),2, jjj*2)
                        plot(squeeze(coeffs_(1:End,k,j)/y_max));hold on
                        %plot(squeeze(true_time(1:End,k,j)));hold on
                        plot([1 End],[0 0],'k-')
                        % plot(squeeze(coeffs_(1:End,k,j))/y_max);
                        plot([t-order t-order],[-1 1],'k-')
                        
                        xlim([0 End]); ylim([-1 1])
                        ylabel([num2str(k),'<-',List_{j}])
                        hold off
                        jjj = jjj + 1 ;
                    end
                end
                
                % motion
                subplot(1,2,1)
                for k = 1:K
                    marker = 'o';
                    xy = pos(t,:,k) ;
                    if t <= duration
                        xy_long = pos(1:t,:,k) ;
                    else; xy_long = pos(t-duration:t,:,k) ;
                    end
                    ms = 12 ; lw = 1 ;
                    if mod(k,5) == 0 ; clr = 'r' ;
                    elseif mod(k,5) == 1; clr = 'g' ;
                    elseif mod(k,5) == 2; clr = 'b' ; %
                    elseif mod(k,5) == 3; clr = 'k' ;
                    elseif mod(k,5) == 4; clr = 'm' ;
                    end
                    if contains(filenames{f},'peregrine')
                        plot3(xy(1),xy(2),xy(3),marker,'markersize',ms,'linewidth',lw,'color',clr); hold on;
                        plot3(xy_long(:,1),xy_long(:,2),xy_long(:,3),'-','color',clr); hold on;
                        %text(xy(1),xy(2),xy(3),num2str(k));
                        text(xy(1),xy(2),xy(3),num2str(k));
                    else
                        plot(xy(1),xy(2),marker,'markersize',ms,'linewidth',lw,'color',clr); hold on;
                        plot(xy_long(:,1),xy_long(:,2),'-','color',clr); hold on;
                        %text(xy(1),xy(2),xy(3),num2str(k));
                        text(xy(1),xy(2),num2str(k));
                    end
                end
                hold off
                
                axis equal
                if 0 % contains(filenames{f},'sula')
                elseif contains(filenames{f},'peregrine')
                    VIEW = [-10 40]
                    set(gca,'xlim',[min_xy(f,1) max_xy(f,1)],'ylim',[min_xy(f,2) max_xy(f,2)],...
                        'zlim',[min_xy(f,3) max_xy(f,3)],'View', [-10 40]) ; % [-84,12]);%
                elseif contains(filenames{f},'mice')
                    set(gca,'xlim',[min_xy(1,1) max_xy(1,1)],'ylim',[min_xy(1,2) max_xy(1,2)])
                else
                    set(gca,'xlim',[min_xy(f,1) max_xy(f,1)],'ylim',[min_xy(f,2) max_xy(f,2)])
                end
                xlabel('x')
                ylabel('y')
                zlabel('z')
                %axis off
                box off
                
                title([filenames{f},sprintf(', Frame %d (%dHz)', t, floor(Fs))],'Fontsize',8) ; % Video %04d n,
                
                mov(nn)= getframe(gcf);  % mov‚ÌƒCƒ“ƒfƒbƒNƒX
                drawnow
                
                nn = nn+1;
                
            end
            writeVideo(v,mov)
            close(v)
        end
    end
end




