
import logging
from logging import handlers
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射
    def __init__(self,filename,level='info',backCount=3,fmt='%(levelname)s: %(message)s - %(asctime)s - [line:%(lineno)d]'):

        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式

        self.logger.addHandler(sh)  # 把对象加到logger里
        #self.logger.addHandler(th)
log = Logger('all.log',level='debug')
log.logger.debug('debug')




class Logger1(object):
	level_relations = {
	'debug':logging.DEBUG,
	'info':logging.INFO,
	'warning':logging.WARNING,
	'error':logging.ERROR,
	'crit':logging.CRITICAL
	}
	def __init__(self, log_path, level='info', backCount=0,fmt='%(levelname)s: %(message)s - %(asctime)s - [line:%(lineno)d]'):
		log_path = log_path[0]
		filename = log_path.split("\\")[-1].split(".")[0]
		self.logger = logging.getLogger(filename)
		format_str = logging.Formatter(fmt)
		self.logger.setLevel(self.level_relations.get(level))
		sh = logging.StreamHandler()  # 往屏幕上输出
		sh.setFormatter(format_str)
		#th = logging.FileHandler(log_path, encoding='UTF-8')
		#th.setFormatter(format_str)  # 设置文件里写入的格式
		self.logger.addHandler(sh)  # 把对象加到logger里
		#self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger1('all.log',level='debug')
    log.logger.debug('debugqq')
class TempEX():
	def __init__(self):
		self.debug=False
		self.count = 0
		self.json_path='./para.json'

		with open(self.json_path, 'r', encoding='UTF-8') as f:
			para_dict = json.load(f)

		if para_dict['log']=='True' or para_dict['log']=='1':
			self.log=True
		else:
			self.log=False
		if para_dict['save_all_frame']=='True' or para_dict['save_all_frame']=='1':
			self.save_all_frame=True
		else:
			self.save_all_frame=False
		if para_dict['save_key_frame']=='True' or para_dict['save_key_frame']=='1':
			self.save_key_frame=True
		else:
			self.save_key_frame=False
		if para_dict['plot_temp']=='True' or para_dict['plot_temp']=='1':
			self.plot_temp=True
		else:
			self.plot_temp=False
		if para_dict['plot_score']=='True' or para_dict['plot_score']=='1':
			self.plot_score=True
		else:
			self.plot_score=False
		if para_dict['save_dt_frame']=='True' or para_dict['save_dt_frame']=='1':
			self.save_dt_frame=True
		else:
			self.save_dt_frame=False

		if para_dict['save_last_frame']=='True' or para_dict['save_last_frame']=='1':
			self.save_last_frame=True
		else:
			self.save_last_frame=False

		self.dpi=int(para_dict['dpi'])
		self.dt = float(para_dict['dt'])
		num = str(self.dt)
		decimal = num.split('.')[1]
		if decimal == '0':
			self.len_num = 0
		else:
			self.len_num = len(decimal)




		self.vfolder = para_dict['video_folder']
		self.frame_suffix=para_dict['frame_suffix']


		self.out_folder = para_dict['out_folder']
		self.frame_folder = para_dict['frame_folder']
		self.mkdir()


		video_list = []
		self.video_list = TempEX.scanner_video(self.vfolder, video_list)
		print("find %d videos in %s: %s"%(len(video_list),self.vfolder,str(video_list)))
		self.video_name_list=[]

		self.xlsxpath_list=[]
		self.frame_folder_list=[]
		self.out_path_list=[]
		self.save_frame_flag=False
		if self.dt==0:
			if self.save_all_frame or self.save_key_frame:
				self.save_frame_flag = True
		else:
			if self.save_dt_frame:
				self.save_frame_flag=True
		for vi in video_list:
			viname=vi.split("\\")[-1].split('.')[0]
			self.video_name_list.append(viname)
			xlsxname=viname+'.xlsx'
			if self.save_frame_flag:
				out_path=os.path.join(self.out_folder,viname)
				path = os.path.join(out_path, self.frame_folder)
				self.frame_folder_list.append(path)
				if not os.path.exists(path):
					os.makedirs(path)
			else:
				out_path=self.out_folder
			self.out_path_list.append(out_path)
			xlsxpath = os.path.join(out_path, xlsxname)
			self.xlsxpath_list.append(xlsxpath)

		ROI_string=para_dict['ROI']
		#[[5: 50, 5: 130], [0: 45, 0: 90]]
		pattern = re.compile(r"\d+:\d+")
		res = re.findall(pattern, ROI_string)
		self.x1=[]
		self.x2 = []
		self.y1 = []
		self.y2 = []
		for i in range(int(len(res)/2)):
			yy=res[2*i]
			self.y1.append(int(yy.split(":")[0]))
			self.y2.append(int(yy.split(":")[1]))
			xx = res[2 * i+1]
			self.x1.append(int(xx.split(":")[0]))
			self.x2.append(int(xx.split(":")[1]))

		self.prefer_ROI=int(para_dict['prefer_ROI'])
		self.ROI_indexs=list(range(len(self.y1)))
		#self.ROI_indexs.remove(self.prefer_ROI)
		#self.ROI_indexs=[self.prefer_ROI]+self.ROI_indexs








		self.model_name=para_dict['model_name']
		self.ocr = CnOcr(det_model_name=self.model_name)  # 所有参数都使用默认值
		print('use CnOcr model %s'%self.model_name)
		self.cdt=None




	def mkdir(self):
		if not os.path.exists(self.vfolder):
			os.makedirs(self.vfolder)
		if not os.path.exists(self.out_folder):
			os.makedirs(self.out_folder)


	@staticmethod
	def scanner_video(inputSrc, video_list):
		file_list = os.listdir(inputSrc)
		for file in file_list:
			curr_file = os.path.join(inputSrc, file)

			# 递归实现
			if (os.path.isdir(curr_file)):
				scanner_video(curr_file, video_list)
			else:
				curr_file_name = curr_file.split(".")
				curr_file_type = curr_file_name[len(curr_file_name) - 1]

				if curr_file_type == "mp4" or curr_file_type == "avi" or curr_file_type == "wmv" or curr_file_type == "vm4":
					video_list.append(curr_file)
		return video_list





	def run(self):
		self.vcount=0
		for vpath in self.video_list:
			self.time_tamp = []
			self.temps = []
			self.score = []
			self.temp_old = ''
			self.count=0
			capture = cv2.VideoCapture(vpath)

			frames_total = capture.get(7)
			logname = os.path.join(self.out_path_list[self.vcount], self.video_name_list[self.vcount] + '.log'),
			#log = Logger(logname, level='debug')
			log = Logger('all.log', level='debug')

			log.logger.debug('***************load video %d in %s ********************************'%(self.vcount+1,vpath))
			if capture.isOpened():
				while True:
					success, frame = capture.read()
					if not success:
						self.logger.info('--------------------------finish video process--------------------------')
						self.write_xlsx()
						if self.dt==0:
							if self.save_all_frame:
								self.logger.info("save all frames into %s"%self.frame_folder_list[self.vcount])
							else:
								if self.save_key_frame:
									self.logger.info("save key frames into %s" % self.frame_folder_list[self.vcount])
						else:
							if self.save_dt_frame:
								self.logger.info("save dt=%f s frames into %s" % (self.dt,self.frame_folder_list[self.vcount]))


						self.plots()

						break
					else:
						time_s = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
						self.count = self.count + 1
						if self.debug:
							if self.count==15816:
								temp=self.frame_pro(frame,time_s)
								if self.save_all_frame or self.save_key_frame:
									self.save_frame(frame, temp, time_s)
						else:
							if self.dt == 0:
								temp=self.frame_pro(frame, time_s)
								if self.save_all_frame or self.save_key_frame:
									self.save_frame(frame, temp, time_s)
							else:
								cdt = time_s // self.dt
								if cdt == self.cdt:
									if self.count==frames_total:
										if self.save_last_frame:
											cflag=True
											self.logger.info("----- process last frame")
										else:
											cflag=False
									else:
										cflag=False
								else:
									cflag=True
								if cflag:
									temp=self.frame_pro(frame, time_s)
									if self.save_dt_frame:
										self.save_frame(frame, temp, time_s)
									self.cdt = cdt




			else:
				self.logger.warning('WARNING! %s loading failed'% vpath)
			self.vcount = self.vcount + 1
		plt.show()
	def temp_read(self,frame):
		#plt.imshow(frame)
		#plt.show()
		#print(self.count)
		self.ROI_indexs.remove(self.prefer_ROI)
		ROI_indexs = [self.prefer_ROI] + self.ROI_indexs
		self.ROI_indexs=ROI_indexs


		temp_f=0
		score=0
		flag=True
		scores=[]
		temps=[]
		i=0
		while flag and i<len(ROI_indexs):
			index_ROI=ROI_indexs[i]
			if i>0:
				self.logger.warning("-- change ROI to ROI%d"%index_ROI)
			i=i+1
			ROI=frame[self.y1[index_ROI]:self.y2[index_ROI], self.x1[index_ROI]:self.x2[index_ROI]]
			ROI= cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
			ret2, ROI = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			try:
				outs = self.ocr.ocr(ROI)
				if outs is None:
					self.logger.warning("--waring!! ocr result None, switch ROI")
					continue
				else:
					num = len(outs)
			except:
				self.logger.warning("--waring!! ocr error, switch ROI")
				plt.imshow(ROI)
				plt.show()
				continue
			if num==0:
				self.logger.warning("--waring!! ocr result Null, swicth ROI")
				continue
			else:
				if num>1:
					self.logger.info("-- find %d ocr results"%num)
				for ii in range(num):
					out=outs[ii]
					if ii>0:
						self.logger.info("--serach ocr results%d: %s" % (num, out))
					temp = out['text']
					if temp is None:
						self.logger.warning("--waring!! temp result none, swicth ocr result")
						continue
					else:
						pattern = re.compile(r"\d+\.\d")
						res = re.findall(pattern, temp)
						if len(res)>0:
							temp_f = float(temp)
							score = out['score']
							self.prefer_ROI = index_ROI
							if ii > 0:
								self.logger.info("** ocr result is number, change ROI to %d" % self.prefer_ROI)
							flag = False
							break
						else:
							self.logger.warning("--warning!! ocr result is not full correct number, try to find number in it")
							pattern = re.compile(r"\d+.*\d")
							res=re.findall(pattern,temp)
							if len(res)>0:
								for temp_string in res:
									pattern = re.compile(r"\d+")
									temp_re = re.findall(pattern, temp_string)
									if len(temp_re)>1:
										temp_string=temp_re[0]+'.'+temp_re[1]
									else:
										temp_string=temp_re[0]
										self.logger.warning("--warning!! ocr find 1 int number, please enlarge ROI")

									temp_fs = float(temp_string)
									self.logger.info("--find number %.2f in %s"%(temp_fs,temp_string))
									temps.append(temp_fs)
									scores.append(out['score'])
							else:
								self.logger.warning("--warning!! no number in ocr result, swicth ocr result")
								continue


		if flag:
			if len(temps)>0:
				index=scores.index(max(scores))
				temp_f=temps[index]
				score=scores[index]
				self.logger.warning("--------------attention!! %d frame ocr is not credible,please check!" % self.count)
			else:
				self.logger.error("--------------error!! %d frame ocr failed,please check!"%self.count)
				temp_f=0
				score=0

		return temp_f,score

	def frame_pro(self,frame,time_s):
		temp_f,score=self.temp_read(frame)

		self.logger.info("frame: %d, time:%f ms: temp:%f, score:%f "%(self.count,time_s*1000,temp_f,score))
		self.time_tamp.append(time_s)
		self.temps.append(temp_f)
		self.score.append(score)
		return temp_f

	def save_frame(self,frame,temp,time_s):
		filename=str(self.count)+'_'+str(temp)+self.frame_suffix
		filepath=os.path.join(self.frame_folder_list[self.vcount],filename)
		if self.dt==0:
			if self.save_all_frame:
				cv2.imwrite(filepath, frame)
			else:
				if self.save_key_frame:
					if temp==self.temp_old:
						pass
					else:
						cv2.imwrite(filepath, frame)
					self.temp_old=temp
		else:
			form="{:."+str(self.len_num)+"f}"
			tim_str=form.format(time_s)

			filename = tim_str + 's_' + str(temp) + self.frame_suffix
			filepath = os.path.join(self.frame_folder_list[self.vcount], filename)
			cv2.imwrite(filepath, frame)
			self.logger.info("save %s s frame"%(tim_str))





	def plots(self):
		out_path=self.out_path_list[self.vcount]
		name=self.video_name_list[self.vcount]
		#plt.ion()
		if self.plot_temp:
			#fcount = self.fcount + 1
			#self.fcount=fcount
			title="Time-Temperature"
			ptitle=name+" "+title
			plt.figure(ptitle)
			plt.plot(self.time_tamp,self.temps,'k')
			plt.xlabel("Time / s")
			plt.ylabel("Temperature / $^\circ$C")
			plt.title(title)
			fpath=os.path.join(out_path,ptitle+self.frame_suffix)
			plt.savefig(fpath,dpi=self.dpi,bbox_inches='tight')
			self.logger.info("save %s in %s dpi=%d"%(ptitle,fpath,self.dpi))

		if self.plot_score:
			#fcount = self.fcount + 1
			#self.fcount=fcount
			title = "Time-Temperature-Score"
			ptitle = name + " " + title
			x=self.time_tamp
			y1=self.temps
			y2=self.score
			fig=plt.figure(ptitle)
			ax1 = fig.add_subplot(111)
			ax1.plot(x, y1, 'r', label="Temperature");
			ax1.legend(loc=1)
			ax1.set_ylabel('Temperature / $^\circ$C');
			ax1.set_xlabel('Time / s');

			ax2 = ax1.twinx()  # this is the important function
			ax2.plot(x, y2, 'g', label="Score")
			ax2.legend(loc=2)
			#ax2.set_xlim([0, np.e]);
			ax2.set_ylabel('Score');
			ax2.set_xlabel('Time / ms');
			plt.title(title)
			fpath = os.path.join(out_path, ptitle + self.frame_suffix)
			fig.savefig(fpath,dpi=self.dpi,bbox_inches='tight')
			self.logger.info("save %s in %s dpi=%d" % (ptitle, fpath, self.dpi))


	def write_xlsx(self):
		Data=[self.time_tamp,self.temps,self.score]
		self.logger.info('write result into %s'%self.xlsxpath_list[self.vcount])
		TempEX.xw_toExcel(Data, self.xlsxpath_list[self.vcount])
	@staticmethod
	def xw_toExcel(data, fileName):  # xlsxwriter库储存数据到excel
		workbook = xw.Workbook(fileName)  # 创建工作簿
		worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
		worksheet1.activate()  # 激活表
		title = ['time / s', 'temprature','score']  # 设置表头
		worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
		i = 2  # 从第二行开始写入数据
		for j in range(len(data[0])):
			insertData = [data[0][j], data[1][j], data[2][j]]
			row = 'A' + str(i)
			worksheet1.write_row(row, insertData)
			i += 1
		workbook.close()  # 关闭表