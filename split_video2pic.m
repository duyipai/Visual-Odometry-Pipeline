obj = VideoReader('preview.mp4')
obj_numberofframe = obj.NumberOfFrame;%读取总的帧数
% obj_height = obj.Height;%读取视频帧高度

% 读取前n帧
for k = 1 : obj_numberofframe 
     frame = read(obj,k);%读取第几帧
    % imshow(frame);%显示帧   need to create the folder first
      imwrite(frame,strcat('spit_pic/',num2str(k),'.jpg'),'jpg');% 保存帧
end