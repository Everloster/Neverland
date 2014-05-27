/**
 * Copyright (c) 2013 NearMe, All Rights Reserved.
 * FileName:IDatagramService.java
 * ProjectName:NearmeBaseToolsJ
 * PackageName:com.nearme.base.netty.datagram
 * Create Date:2013-8-22
 * History:
 *   ver	date	  author		desc	
 * ────────────────────────────────────────────────────────
 *   1.0	2013-8-22	  80036381		
 *
 * 
*/

package com.nearme.base.netty.datagram;

import java.net.SocketAddress;

/**
 * ClassName:IDatagramService
 * Function: TODO ADD FUNCTION
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2013-8-22  下午6:16:43
 */
public interface IDatagramService {
	void handleMessage(byte[] msg, SocketAddress socketAddress);
	
	void handleError(Throwable t);
}

