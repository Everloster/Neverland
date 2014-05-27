/**
 * RequestUtil.java
 * com.nearme.base.http
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-2-11 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.http;

import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.HashMap;
import java.util.Map;

import org.jboss.netty.handler.codec.http.HttpRequest;

import com.nearme.base.netty.common.DataHelper;
import com.oppo.base.common.OConstants;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:RequestUtil <br>
 * Function: HttpRequest(netty)参数处理 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-2-11  下午06:45:09
 */
public class RequestUtil {
	/**
	 * 从url解析得到所有参数
	 * @param 
	 * @return
	 */
	public static Map<String, String> getParameterMap(String url) {
		Map<String, String> paramMap = new HashMap<String, String>();
		int pos = (null == url) ? -1 : url.indexOf('?');
		if(pos < 0) {
			return paramMap;
		}
		
		String inputString = url.substring(pos + 1);
		
		if(!OConstants.EMPTY_STRING.equals(inputString)) {
			//解析字符串得到各个值并发到attribute中
			String[] splitTextArray = StringUtil.split(inputString, '&');
			int length = splitTextArray.length;
			for(int i = 0; i < length; i++) {
				String splitText = splitTextArray[i];
				int index = splitText.indexOf('=');
				if(index > 0) {
					String key = splitText.substring(0, index);
					String value = splitText.substring(index + 1);
					try {
						value = URLDecoder.decode(value, OConstants.DEFAULT_ENCODING);
					} catch(Exception ex) {}
					paramMap.put(key, value);
				}
			}
		}
		
		return paramMap;
	}
	
	/**
	 * 从request中获取所有的参数
	 * @param 
	 * @return
	 */
	public static Map<String, String> getParameterMapFromPost(HttpRequest request) throws UnsupportedEncodingException {
		String content = getPostStringFromRequest(request, null);
		
		return getParameterMap(content);
	}
	
	/**
	 * 从HttpRequest中获取post数据
	 * @param request
	 * @param encoding 转换成字符时使用的编码，如果为null，则会使用默认编码
	 * @return
	 * @throws UnsupportedEncodingException 
	 */
	public static String getPostStringFromRequest(HttpRequest request, String encoding) throws UnsupportedEncodingException {
		byte[] data = getPostBytesFromRequest(request);
		
		if(StringUtil.isNullOrEmpty(encoding)) {
			encoding = OConstants.DEFAULT_ENCODING;
		}
		
		return new String(data, encoding);
	}
	
	/**
	 * 从HttpRequest中获取post数据
	 * @param 
	 * @return
	 */
	public static byte[] getPostBytesFromRequest(HttpRequest request) {
		return DataHelper.getBytesFromChannelBuffer(request.getContent());
	}
	
	public static void main(String[] args) {
		getParameterMap("");
		getParameterMap("?");
		getParameterMap("?as");
	}
}

