/**
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 * FileName:HttpUtil.java
 * ProjectName:OStatistic
 * PackageName:com.oppo.statictis.http
 * Create Date:2011-4-6
 * History:
 *   ver	date	  author		desc	
 * ────────────────────────────────────────────────────────
 *   1.0	2011-4-6	  80036381		
 *
 * 
*/

package com.oppo.base.http;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URLDecoder;

import javax.servlet.http.HttpServletRequest;

import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:HttpUtil
 * Function: Http操作工具类
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-4-6  下午07:09:41
 */
public class HttpUtil {
	/**
	 * 未知IP的定义
	 */
	public static final String UNKNOWN_IP = "unknown";
	
	private static final String INPUT_STRING_ATTR = "inputStringAttribute";

	/**
	 * 获取客户端IP
	 * @param request
	 * @return
	 */
	public static String getClientIP(HttpServletRequest request) {
		//如果有反向代理，则取头部x-forwarded-for中的值
		String ip = request.getHeader("x-forwarded-for");    
		
		//无反向代理，则取代理IP
		if(StringUtil.isNullOrEmpty(ip) || ip.equalsIgnoreCase(UNKNOWN_IP)) {
			ip = request.getHeader("Proxy-Client-IP");
		}
		
		//未获取到代理IP，则取
		if(StringUtil.isNullOrEmpty(ip) || ip.equalsIgnoreCase(UNKNOWN_IP)) {
			ip = request.getHeader("WL-Proxy-Client-IP");
		}
		
		if(StringUtil.isNullOrEmpty(ip) || ip.equalsIgnoreCase(UNKNOWN_IP)) {
			ip = request.getRemoteAddr();
		}
		
		return ip;
	}
	
	/**
	 * 获取站点路径,包括域名+端口+根目录
	 * @param request
	 * @return
	 */
	public static String getWebsite(HttpServletRequest request) {
		return getWebSite(request, true);
	}
	
	/**
	 * 获取站点路径
	 * @param request
	 * @param needPort 是否需要端口
	 * @return
	 */
	public static String getWebSite(HttpServletRequest request, boolean needPort) {
		StringBuilder urlBuilder = new StringBuilder();
		//设置协议
		if(request.isSecure()) {
			urlBuilder.append("https://");
		} else {
			urlBuilder.append("http://");
		}
 
		//设置域名
		urlBuilder.append(request.getServerName());

		if(needPort) {
			//设置端口
			int port = request.getServerPort();
	        if(port != 80 && port != 443) {
	        	urlBuilder.append(":").append(port);
	        }
		}
        
        //站点根目录
        urlBuilder.append(request.getContextPath());

        return urlBuilder.toString();
	}
	
	public static boolean isPost(HttpServletRequest request) {
		return "POST".equalsIgnoreCase(request.getMethod());
	}
	
	public static boolean isGet(HttpServletRequest request) {
		return "GET".equalsIgnoreCase(request.getMethod());
	}
	
	/**
	 * 从request中获取传入的字符串
	 * @param request
	 * @return
	 */
	public static String getInputString(HttpServletRequest request) throws IOException {
		return getInputString(request, false);
	}
	
	/**
	 * 从request中获取传入的字符串
	 * @param request
	 * @param setAttribute 是否将参数解析后存入attribute中,当参数为post上来的时候设置为true,
	 * 			在获取参数值时,用getAttribute代替getParameter,数据提交时必须为utf-8
	 * @return
	 */
	public static String getInputString(HttpServletRequest request, boolean setAttribute) throws IOException {
		//先尝试从attribute中获取
		Object attribute = request.getAttribute(INPUT_STRING_ATTR);
		if(null != attribute) {
			return attribute.toString();
		}
		
		//从InputStream中获取
		InputStream inputStream = request.getInputStream();
		ByteArrayOutputStream baos = new ByteArrayOutputStream(inputStream.available());
		int readLen;
		
		int bufferLen = 1024;
		byte[] data = new byte[bufferLen];
		while((readLen = inputStream.read(data, 0, bufferLen)) != -1) {
			baos.write(data, 0, readLen);
		}

		String inputString = baos.toString(OConstants.DEFAULT_ENCODING);
		//将获取到的值设置到attribute中，供下次调用
		if(!OConstants.EMPTY_STRING.equals(inputString)) {
			request.setAttribute(INPUT_STRING_ATTR, inputString);
			
			if(setAttribute) {
				//解析字符串得到各个值并发到attribute中
				String[] splitTextArray = StringUtil.split(inputString, '&');
				int length = splitTextArray.length;
				for(int i = 0; i < length; i++) {
					String splitText = splitTextArray[i];
					int index = splitText.indexOf('=');
					if(index > 0) {
						String key = splitText.substring(0, index);
						String value = splitText.substring(index + 1);
						request.setAttribute(key, URLDecoder.decode(value, OConstants.DEFAULT_ENCODING));
					}
				}
			}
		}
		return inputString;
	}
	
	public static byte[] getInputBytes(HttpServletRequest request) throws IOException  {
		//从InputStream中获取
		InputStream inputStream = request.getInputStream();
		ByteArrayOutputStream baos = new ByteArrayOutputStream(inputStream.available());
		int readLen;
		
		int bufferLen = 1024;
		byte[] data = new byte[bufferLen];
		while((readLen = inputStream.read(data, 0, bufferLen)) != -1) {
			baos.write(data, 0, readLen);
		}
		
		return baos.toByteArray();
	}
}

