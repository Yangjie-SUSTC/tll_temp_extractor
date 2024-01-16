第一次运行需要双击install_pkg.bat来安装需要的包
之后每次运行双击run.bat或者当前路径打开cmd，输入 python video_temp_extractor.py回车
因为需要加载模型，所以需要等待5s才会开始执行
会自动处理完所有在video文件夹下的视频，并将结果输出到out文件夹中
输出结果中score为识别结果的得分（置信度），取值范围为 [0, 1]；得分越高表示越可信
所有配置都在para.json中定义，该文件需存放与exe文件相同路径中

文件结构如下
para.json
video_temp_extractor.exe
    -video
        视频1.mp4
        视频2.mp4
    -out
        -视频1
            视频1.xls
            -frame
                各帧.jpg
        -视频2
            视频2.xls
            -frame
                各帧.jpg

配置说明如下：
    "dt": "2",                              // 间隔的时间，单位s，为0的话不执行间隔一段时间保存
    "save_dt_frame": "1",                   // 是否保存指定时间间隔的帧，1保存，0不保存
    "save_last_frame": "1",                 // 在指定时间间隔的情况下，当最后一帧不是最后一个时间点时是否处理，1处理，0不处理

    "ROI": "[[0:45, 0:90],[5:50,5:130]]",   // 视频中显示温度区域的坐标序列，值为[起始行:终止行，起始列:终止列]]，会在给定的序列中选择最合适的ROI
    "video_folder" : ".\\video",            // 视频放置目录，默认为当前路径下video文件夹，支持mp4， avi， wmv， vm4格式
    "out_folder" : ".\\out",                // 输出结果目录，默认为当前路径下out文件夹
    "save_key_frame": "1",                  // 是否保存关键帧，丢掉温度不变的帧，1保存，0不保存,只在dt为时有效
    "save_all_frame": "0",                  // 是否保存全部帧，1保存，0不保存,只在dt为时有效
    "frame_folder" : "frmae",               // 视频帧保存的路径名，默认为out\视频名\frame
    "frame_suffix": ".jpg",                 // 保存的帧后缀，默认为jpg
    "plot_temp":"1",                        // 是否绘制温度随时间曲线，1绘制，0不绘制
    "plot_score":"1",                       // 是否绘制温度，分数，随时间曲线，
    "dpi":"300",                            // 绘制的曲线保存的分辨率，默认为300dpi

    "log"     : "0",                        // 是否输出日志，默认为0，直接打印
    "consol_level"  : "info",               // 输出窗口日志等级，'debug'，'info'，'warning'，'error'，'crit'依次递增，默认info，只输出必要和异常结果，输出的日志文件会记录所有等级信息
    "model_name":"naive_det"                // 选择的模型名称，默认naive_det，不要更改
    "fps": "7"                              // 默认的视频帧率，用来估计处理时间
