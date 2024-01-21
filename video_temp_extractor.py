import logging
import time
from colorama import init, Fore, Back, Style
import cv2
import os
import matplotlib
import re

from matplotlib import pyplot as plt
import json
from cnocr import CnOcr
import xlsxwriter as xw

import numpy as np

import pytesseract
matplotlib.use('TkAgg')




class MyLogger(object):
	level_relations = {
	'debug':logging.DEBUG,
	'info':logging.INFO,
	'warning':logging.WARNING,
	'error':logging.ERROR,
	'crit':logging.CRITICAL
	}
	def __init__(self, log_path, log,level='info',fmt='%(levelname)s: %(message)s - %(asctime)s - [line:%(lineno)d]'):
		log_path = log_path[0]
		log_path =log_path [:-4]+'_'+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())+log_path[-4:]
		filename = log_path.split("\\")[-1].split(".")[0]
		self.logger = logging.getLogger(filename)
		self.logger.propagate = False
		format_str = logging.Formatter(fmt)
		self.logger.setLevel(self.level_relations.get('debug'))
		sh = logging.StreamHandler()  # 往屏幕上输出
		sh.setLevel(self.level_relations.get(level))
		sh.setFormatter(format_str)
		self.logger.addHandler(sh)  # 把对象加到logger里

		if log:
			th = logging.FileHandler(log_path, encoding='UTF-8')
			th.setFormatter(format_str)  # 设置文件里写入的格式
			self.logger.addHandler(th)

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

		pytesseract.pytesseract.tesseract_cmd = para_dict['tesseract_cmd'] # your path may be different
		self.max_dif_ratio=float(para_dict['max_dif_ratio'])
		self.old_data = None

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

		print(Fore.GREEN+"find %d videos in %s: %s"%(len(video_list),self.vfolder,str(video_list)))
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
		print(Fore.GREEN+'use CnOcr model %s'%self.model_name)
		self.version=para_dict['version']
		self.consol_level = para_dict['consol_level']
		print('dt:%f s, save dt frame: %s\n'
			  'save last frame %s\n'
			  'save key frame: %s\n'
			  'save all frame: %s\n'
			  'write log: %s' % (self.dt,self.save_dt_frame,self.save_last_frame,self.save_key_frame,self.save_all_frame,self.log))
		print("vesion: %s "%self.version)
		print("consol log level: %s " % self.consol_level)
		print(Fore.WHITE)
		self.cdt=None
		self.r_frame_count=0
		self.fps = int(para_dict['fps'])





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
			Tv1 = time.time()
			self.time_tamp = []
			self.temps = []
			self.score = []
			self.temp_old = ''
			self.count=0
			self.r_frame_count=0
			capture = cv2.VideoCapture(vpath)
			logname = os.path.join(self.out_path_list[self.vcount], self.video_name_list[self.vcount] + '.log'),
			self.logger = MyLogger(logname, self.log, level=self.consol_level).logger

			frames_total = capture.get(7)
			fps=self.fps

			self.logger.info("video encoded in %f FPS"%(fps))
			if self.dt<1/fps:
				self.logger.warning("Too high sample ratio: %f s < dt in video: %f s"%(self.dt,1/fps))


			#log = MyLogger('all.log', level='debug')

			self.logger.info('***************load video %d in %s ********************************'%(self.vcount+1,vpath))
			self.logger.info("** processing .... please wait %f s"%(frames_total*0.025/(self.dt*fps)))
			if capture.isOpened():
				while True:
					success, frame = capture.read()
					if not success:
						self.logger.info('-------------------------- finish video process--------------------------')
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
							if self.count==12374:
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
											self.logger.info("-- process last frame")
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
				self.logger.warning('%s loading failed'% vpath)
			self.vcount = self.vcount + 1
			Tv2 = time.time()
			self.logger.info(
				'******************************************** %d / %d frames, used %s s *********************************' % (
				self.r_frame_count,frames_total, (Tv2 - Tv1)))
			print("\n\n")

		plt.show()

	def draw_ROI(self,frame,ROI, ROI_cor):
		plt.figure()
		plt.subplot(121)
		plt.imshow(frame)
		ax = plt.gca()
		# 默认框的颜色是黑色，第一个参数是左上角的点坐标
		# 第二个参数是宽，第三个参数是长
		ax.add_patch(plt.Rectangle((ROI_cor[0],ROI_cor[2]), ROI_cor[3]-ROI_cor[2],ROI_cor[1]-ROI_cor[0], color="red", fill=False, linewidth=2))
		plt.title("%d frame, ROI:%s"%(self.count,ROI_cor))
		plt.subplot(122)
		plt.imshow(ROI)
		plt.title("ROI")
		plt.show()
	def selct_best(self,datas,ocr_data,ocr_sore,old_data):
		if old_data is None:
			datas.append(ocr_data)
			nums=[]
			for i in datas:
				num=datas.count(i)
				nums.append(num)
			temp=datas[np.argmax(nums)]
			num = nums[np.argmax(nums)]
			if temp == ocr_data:
				score = ocr_sore
				self.logger.info("++++++++++ %d frame final robust ocr result: %.4f, score: %.4f, repeat num: %d" % (
					self.count, temp, score, num))
			else:
				score = num/len(datas)
				self.logger.info("++++++++++ %d frame final  ocr result: %.4f, score: %.4f, repeat num: %d" % (
					self.count, temp, score, num))


		else:
			score=0
			if ocr_data in datas:
				dif_ratio=abs(ocr_data-old_data)/old_data
				if dif_ratio<self.max_dif_ratio:
					temp=ocr_data
					score=1
					self.logger.info("++++++++++ %d frame final robust ocr result: %.4f, score: %.4f, dif_ratio: %.4f" % (self.count, temp, score,dif_ratio))
			if not score==1:
				datas.append(ocr_data)
				diff=[i-old_data for i in datas]
				temp=datas[np.argmin(diff)]
				if temp==ocr_data:
					score=ocr_sore
				else:
					score=0.5
				dif_ratio = abs(temp - old_data) / old_data
				if dif_ratio<self.max_dif_ratio:
					self.logger.info("---------- %d frame final closest ocr result: %.4f, score: %.4f, dif_ratio: %.4f" % (self.count, temp, score,dif_ratio))
				else:
					score = 0
					self.logger.warning(
						"********** attention!! %d frame abnormal ocr result: %.4f, score: %.4f,dif_ratio: %.4f >  %.4f, please check" % (self.count, temp, score,dif_ratio,self.max_dif_ratio))
		return temp, score

	def tesocr_model(self,frame):


		self.ROI_indexs.remove(self.prefer_ROI)
		ROI_indexs = [self.prefer_ROI] + self.ROI_indexs
		self.ROI_indexs = ROI_indexs


		flag = True
		scores = []

		temps = []
		i = 0
		while flag and i < len(ROI_indexs):
			index_ROI = ROI_indexs[i]
			if i > 0:
				self.logger.warning("-- change ROI to ROI%d" % index_ROI)
			i = i + 1
			ROI_cor = [self.y1[index_ROI], self.y2[index_ROI], self.x1[index_ROI], self.x2[index_ROI]]
			ROI = frame[ROI_cor[0]:ROI_cor[1], ROI_cor[2]:ROI_cor[3]]
			ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
			ret2, ROI = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

			for psm_num in [6,7,8,13]:
				config = '--psm '+str(psm_num)+' --oem 3 -c tessedit_char_whitelist=.0123456789'
				try:
					data = pytesseract.image_to_string(ROI, lang='eng',config=config)[:-1]
					self.logger.debug("-- %d frame pytesseract ocr --psm %d result: %s" % (self.count, psm_num, data))
					pattern = re.compile(r"^\d+\.\d")
					res = re.findall(pattern, data)
					data=res[0]

					temps.append(float(data))
					scores.append(1)

				except:
					self.logger.warning("ocr error, switch psm")
					scores.append(0)
					if self.debug:
						self.draw_ROI(ROI, ROI, ROI_cor)


		return temps, scores

	def cnocr_read(self,frame,time_s):
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
		intflags=[]
		temps=[]
		i=0
		while flag and i<len(ROI_indexs):
			index_ROI=ROI_indexs[i]
			if i>0:
				self.logger.warning("-- change ROI to ROI%d"%index_ROI)
			i=i+1
			ROI_cor=[self.y1[index_ROI],self.y2[index_ROI], self.x1[index_ROI],self.x2[index_ROI]]
			ROI=frame[ROI_cor[0]:ROI_cor[1], ROI_cor[2]:ROI_cor[3]]
			ROI= cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
			ret2, ROI = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


			try:
				outs = self.ocr.ocr(ROI)
				if outs is None:
					self.logger.warning("cnocr result None, switch ROI")
					if self.debug:
						self.draw_ROI(ROI, ROI, ROI_cor)
					continue
				else:
					num = len(outs)
			except:
				self.logger.warning("cnocr error, switch ROI")
				if self.debug:
					self.draw_ROI(ROI,ROI,ROI_cor)
				continue
			if num==0:
				self.logger.warning("cnocr result Null, swicth ROI")
				if self.debug:
					self.draw_ROI(ROI,ROI,ROI_cor)
				continue
			else:
				if num>1:
					self.logger.info("-- find %d cnocr results"%num)
				for ii in range(num):
					out=outs[ii]
					if ii>0:
						self.logger.info("-- serach cnocr results%d: %s" % (num, out))
					temp = out['text']
					if temp is None:
						self.logger.warning("temp result none, swicth cnocr result")
						if self.debug:
							self.draw_ROI(ROI, ROI, ROI_cor)
						continue
					else:
						pattern = re.compile(r"^\d+\.\d")
						res = re.findall(pattern, temp)
						self.logger.debug("-- %d frame cnocr result: %s" % (self.count,temp))
						if len(res)==1:
								temp_f = float(res[0])
								score = out['score']
								self.prefer_ROI = index_ROI
								if ii > 0:
									self.logger.info("-- cnocr result is full correct number, change ROI to %d" % self.prefer_ROI)
								flag = False
								break

						else:
							self.logger.warning("cnocr result is not full correct number, try to find number in it")
							pattern = re.compile(r"\d+.*\d")
							res=re.findall(pattern,temp)
							if len(res)>0:
								for temp_string in res:
									pattern = re.compile(r"\d+")
									temp_re = re.findall(pattern, temp_string)
									intflag=0
									if len(temp_re)>1:
										sstr=''
										sstr=sstr.join(temp_re[0:-1])
										temp_string=sstr+'.'+temp_re[-1]
									else:
										temp_string=temp_re[0]
										self.logger.warning("cnocr find 1 int number, please enlarge ROI")
										intflag=1

									temp_fs = float(temp_string)
									self.logger.info("-- cnocrfind number %.2f in %s"%(temp_fs,temp_string))
									temps.append(temp_fs)
									scores.append(out['score'])
									intflags.append(intflag)
							else:
								self.logger.warning("no number in cnocr result, swicth ocr result")
								if self.debug:
									self.draw_ROI(ROI, ROI, ROI_cor)
								continue


		if flag:

			if len(temps)>0:
				temps_filtered=[j for i, j in zip(intflags, temps) if i == 1]
				if len(temps_filtered)>1:
					self.logger.warning("--attention!! remove int number, %s ---> %s" % (temps,temps_filtered))
					temps=temps_filtered
					scores=[j for i, j in zip(intflags, scores) if i == 1]

				index=scores.index(max(scores))
				temp_f=temps[index]
				score=scores[index]
				self.logger.warning("find multiple result,temps: %s, scores: %s, choose max score index:%d,temp: %f, score: %f" % (temps,scores,index,temp_f,score))
				self.logger.warning("--attention!! %d frame ocr is not credible,please check!" % self.count)
				self.save_frame(ROI,'attention_'+str(self.count),time_s)
			else:
				self.logger.error("%d frame ocr failed,please check!"%self.count)
				temp_f=0
				score=0
				self.save_frame(frame, 'error_'+str(self.count), time_s)

		return temp_f,score

	def frame_pro(self,frame,time_s):
		#temp_f,score=self.temp_read(frame,time_s)


		temp_f1,score1=self.cnocr_read(frame,time_s)
		temps, scores = self.tesocr_model(frame)
		temp_f, score=self.selct_best(temps,temp_f1,score1,self.old_data)
		self.old_data=temp_f
		self.r_frame_count=self.r_frame_count+1

		self.logger.debug("frame: %d, time:%f ms: temp:%f, score:%f "%(self.count,time_s*1000,temp_f,score))
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
			self.logger.debug("save %s s frame"%(tim_str))





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
		title = ['time / s', 'temperature','score']  # 设置表头
		worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
		i = 2  # 从第二行开始写入数据
		for j in range(len(data[0])):
			insertData = [data[0][j], data[1][j], data[2][j]]
			row = 'A' + str(i)
			worksheet1.write_row(row, insertData)
			i += 1
		workbook.close()  # 关闭表


if __name__ == '__main__':
	T1 = time.time()
	tempv=TempEX()
	try:
		tempv.run()
	except:
		tempv.logger.error("%d frame error, write result into xls"%tempv.count)
		try:
			tempv.write_xlsx()
		except:
			tempv.logger.critical("%d frame cannot write result into xls"%tempv.count)
	T2 = time.time()
	print(Fore.YELLOW+'%d videos,used %s s' % (tempv.vcount,(T2 - T1)))


