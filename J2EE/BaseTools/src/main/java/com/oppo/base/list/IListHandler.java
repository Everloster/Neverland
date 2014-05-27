/**
 * Copyright (c) 2012 NearMe, All Rights Reserved.
 * FileName:IListHandler.java
 * ProjectName:OBaseTool
 * PackageName:com.oppo.base.list
 * Create Date:2012-1-16
 * History:
 *   ver	date	  author		desc	
 * ────────────────────────────────────────────────────────
 *   1.0	2012-1-16	  80054252		
 *
 * 
*/

package com.oppo.base.list;
/**
 * ClassName:IListHandler
 * Function: 指定List中对象在保存至Map对象时的KEY
 * @author   80054252
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-1-16  下午04:41:22
 */
public interface IListHandler<T> {

	/**
	 * 指定List中对象在保存至Map对象时的KEY
	 * 
	 * @param 
	 * @return
	 */
	String getMapKey(T obj);
}

