/**
 * ProxoolUtil.java
 * com.oppo.base.db
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-7-16 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.oppo.base.db;

import java.util.Date;

import org.logicalcobwebs.proxool.ProxoolException;
import org.logicalcobwebs.proxool.ProxoolFacade;
import org.logicalcobwebs.proxool.admin.SnapshotIF;

import com.oppo.base.time.TimeFormat;

/**
 * ClassName:ProxoolUtil <br>
 * Function: Proxool连接池工具类 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-7-16  下午05:47:04
 */
public class ProxoolUtil {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * 获取proxool连接池的状态
	 * @param 
	 * @return
	 */
	public static String getStatus(String alias) throws ProxoolException {
		SnapshotIF snapshot = ProxoolFacade.getSnapshot(alias, true);
		int actConnCnt = snapshot.getActiveConnectionCount();
		int availConnCnt = snapshot.getAvailableConnectionCount();
		long connCnt = snapshot.getConnectionCount();
		int maxConnCnt = snapshot.getMaximumConnectionCount();
		Date dateStarted = snapshot.getDateStarted();
		long refusedCnt = snapshot.getRefusedCount();
		long serverCnt = snapshot.getServedCount();
		
		StringBuilder statusBuilder = new StringBuilder();
		statusBuilder.append("start at:" + TimeFormat.getTimeString(dateStarted));
		statusBuilder.append(",active connection count:").append(actConnCnt);
		statusBuilder.append(",available connection count:").append(availConnCnt);
		statusBuilder.append(",connection count:").append(connCnt);
		statusBuilder.append(",max connection count:").append(maxConnCnt);
		statusBuilder.append(",refuesed count:").append(refusedCnt);
		statusBuilder.append(",server count:").append(serverCnt);
		
		return statusBuilder.toString();
	}
}

