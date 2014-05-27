package com.oppo.base.http;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.oppo.base.common.NumericUtil;
import com.oppo.base.common.StringUtil;
import com.oppo.base.file.FileOperate;


/**
 * 提供服务器下载服务,支持断点续传,主要作用为增加权限判断,防止盗链
 * 如果需要做防盗链或权限判断,请重写isLegal()方法
 * @author 80036381
 *
 */
public class HttpOutput {
	private HttpServletRequest request;
	private HttpServletResponse response;
	private String filePath;				//文件路径
	private boolean isSupportRange;			//是否支持断点续传				

	protected final int BUFF_SIZE = 1024 * 8; //每次8k
	
	public HttpOutput(HttpServletRequest request, HttpServletResponse response) {
		this(request, response, null);
	}
	
	public HttpOutput(HttpServletRequest request, HttpServletResponse response, String filePath) {
		this.request = request;
		this.response = response;
		this.filePath = filePath;
	}

	/**
	 * 服务器信息发送
	 */
	public void output() throws IOException {
		if(!isLegal()) {
			sendError(HttpServletResponse.SC_UNAUTHORIZED);
			return;
		}
		
		File oFile = new File(filePath);
		if(!oFile.exists() || !oFile.isFile() || oFile.isHidden()) {
			sendError(HttpServletResponse.SC_NOT_FOUND);
			return;
		}
		
		//修正开始下载位置及下载总长度
		long total = oFile.length();
		
		initialHeader();

		long startIndex = 0;
		long length = total;
		if(this.isSupportRange) {
			response.setHeader("Accept-Ranges", "bytes"); 
			//如果支持断点续传且有Range，则表示获取文件的一部分
			String range = request.getHeader("Range");
			if(!StringUtil.isNullOrEmpty(range)) { 
				response.setStatus(HttpServletResponse.SC_PARTIAL_CONTENT);
				String[] rangeInfo = StringUtil.split(range.substring(range.indexOf('=') + 1), '-');//bytes=15-30
				startIndex = NumericUtil.parseInt(rangeInfo[0], 0);
				long endIndex = NumericUtil.parseLong(rangeInfo[1], total - 1);
				length = endIndex - startIndex + 1;
				response.setHeader("Content-Range","bytes " + startIndex + "-" + endIndex + "/" + total);  
			}
		}
		
		response.addHeader("Content-Disposition", "attachment;filename=" + oFile.getName());
		response.addHeader("Content-Length", String.valueOf(length));
		
		handleData(startIndex, length);
	}
	
	/**
	 * 初始化header参数
	 */
	protected void initialHeader() {
		response.reset();
		//禁止缓存
		response.setHeader("Cache-Control", "no-store"); 
		response.setHeader("Pragrma", "no-cache"); 
		response.setDateHeader("Expires", 0);
		
		response.setContentType("application/octet-stream");
	}
	
	/**
	 * 输出指定位置开始,指定长度的数据,需要优化请重写此方法
	 * @param startIndex
	 * @param length
	 */
	protected void handleData(long startIndex, long length) throws IOException {
		FileChannel fIn = null;
		try{
			//读取文件
			fIn = new FileInputStream(filePath).getChannel();
			MappedByteBuffer mbb = fIn.map(FileChannel.MapMode.READ_ONLY, startIndex, length);

			//输出
			ServletOutputStream sos = response.getOutputStream();
			byte[] data = new byte[BUFF_SIZE];

			while(length > 0) {//当还有剩余时
				int currLen = length > BUFF_SIZE ? BUFF_SIZE : (int)length;
				mbb.get(data, 0, currLen);
				sos.write(data, 0, currLen);
				length -= BUFF_SIZE;
			}

			sos.flush();
			sos.close();
			sos = null;
		} finally {
			FileOperate.close(fIn);
		}
	}
	
	/**
	 * 判断当前请求是否合法,需要权限判断,请重写此方法
	 * @return
	 */
	protected boolean isLegal() {
		return true;
	}
	
	protected HttpServletRequest getRequest() {
		return request;
	}
	
	protected HttpServletResponse getResponse() {
		return response;
	}
	
	protected void sendError(int statusCode) {
		try	{
			response.sendError(statusCode);
		} catch(Exception e){}
	}
	
	public String getFilePath() {
		return filePath;
	}

	/**
	 * 设置文件路径
	 * @param filePath
	 */
	public void setFilePath(String filePath) {
		this.filePath = filePath;
	}
	
	/**
	 * 设置是否支持断点续传
	 * @param 
	 * @return
	 */
	public boolean isSupportRange() {
		return isSupportRange;
	}

	/**
	 * 设置是否支持断点续传
	 * @param isSupportRange
	 */
	public void setSupportRange(boolean isSupportRange) {
		this.isSupportRange = isSupportRange;
	}
	
	public static void main(String[] args) {
	}
}
