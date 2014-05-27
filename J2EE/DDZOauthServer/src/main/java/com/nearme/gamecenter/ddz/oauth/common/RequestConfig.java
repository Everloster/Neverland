/**
 * RequestConfig.java
 * com.nearme.base.netty.common
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-1-29 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.gamecenter.ddz.oauth.common;

import java.io.Serializable;

/**
 * ClassName:RequestConfig <br>
 * Function: 请求配置 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-29  下午01:59:55
 */
public class RequestConfig implements Serializable {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	private String requestUrl;	//请求的url
	private String handleClass;	//处理的类
	private String description;	//描述
	private String extension0;	//扩展字段0
	private String extension1;	//扩展字段1
	
	/**
	 * 获取请求的url
	 * @return  the requestUrl
	 * @since   Ver 1.0
	 */
	public String getRequestUrl() {
		return requestUrl;
	}
	
	/**
	 * 设置请求的url
	 * @param   requestUrl    
	 * @since   Ver 1.0
	 */
	public void setRequestUrl(String requestUrl) {
		this.requestUrl = requestUrl;
	}
	
	/**
	 * 获取处理的类
	 * @return  the handleClass
	 * @since   Ver 1.0
	 */
	public String getHandleClass() {
		return handleClass;
	}
	
	/**
	 * 设置处理的类
	 * @param   handleClass    
	 * @since   Ver 1.0
	 */
	public void setHandleClass(String handleClass) {
		this.handleClass = handleClass;
	}

	/**
	 * 获取描述
	 * @return  the description
	 * @since   Ver 1.0
	 */
	public String getDescription() {
		return description;
	}

	/**
	 * 设置描述
	 * @param   description    
	 * @since   Ver 1.0
	 */
	public void setDescription(String description) {
		this.description = description;
	}

	/**
	 * 获取扩展字段0
	 * @return  the extension0
	 * @since   Ver 1.0
	 */
	public String getExtension0() {
		return extension0;
	}

	/**
	 * 设置扩展字段0
	 * @param   extension0    
	 * @since   Ver 1.0
	 */
	public void setExtension0(String extension0) {
		this.extension0 = extension0;
	}

	/**
	 * 获取扩展字段1
	 * @return  the extension1
	 * @since   Ver 1.0
	 */
	public String getExtension1() {
		return extension1;
	}

	/**
	 * 设置扩展字段1
	 * @param   extension1    
	 * @since   Ver 1.0
	 */
	public void setExtension1(String extension1) {
		this.extension1 = extension1;
	}
}

