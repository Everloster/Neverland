/**
 * DBConnect.java
 * com.oppo.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-6-8 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.cache;

import org.apache.ibatis.session.ExecutorType;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;

/**
 * ClassName:DBConnect
 * Function: TODO ADD FUNCTION
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-8  上午11:22:21
 */
public class DBConnect extends AbstractCache {
	/**
	 * select one模式
	 */
	public static final int SELECT_TYPE_ONE = 0;
	
	/**
	 * select list模式
	 */
	public static final int SELECT_TYPE_LIST = 1;
	
	private SqlSessionFactory sqlSessionFactory;
	private SqlSession sqlSession;
	private int selectType;
	
	public DBConnect() {
		super();
		
		selectType = SELECT_TYPE_ONE;
	}

	@Override
	public int deleteCache(String cacheId, StorageObject storageObject, long timeout) throws Exception {
		return getSqlSession().delete(storageObject.getStorageId(), storageObject.getStorageObject());
	}

	@Override
	public Object getCache(String cacheId, StorageObject storageObject) throws Exception {
		if(selectType == SELECT_TYPE_ONE) {
			return getSqlSession().selectOne(storageObject.getStorageId(), storageObject.getStorageObject());
		} else if(selectType == SELECT_TYPE_LIST) {
			return getSqlSession().selectList(storageObject.getStorageId(), storageObject.getStorageObject());
		}
		
		return null;
	}

	@Override
	public int insertCache(String cacheId, StorageObject storageObject, long timeout) throws Exception {
		return getSqlSession().insert(storageObject.getStorageId(), storageObject.getStorageObject());
	}

	@Override
	public int updateCache(String cacheId, StorageObject storageObject, long timeout) throws Exception {
		return getSqlSession().update(storageObject.getStorageId(), storageObject.getStorageObject());
	}

	@Override
	public boolean setCache(String cacheId, Object obj, long timeout) {
		//数据访问层不需要设置缓存
		return false;
	}
	
	/**
	 * 开始事务
	 * @param 
	 * @return
	 */
	public void beginTransaction() {
		//释放之前的资源
		this.releaseResource();
		
		//打开一个
		sqlSession = sqlSessionFactory.openSession(ExecutorType.REUSE, false);
	}
	
	/**
	 * 提交事务
	 * @param force 当force为true时,事务一定会提交;<br>
	 * 		当force为false时,只有调用过insert、update、delete才会执行提交
	 * @return
	 */
	public void commitTransaction(boolean force) {
		if(null != this.sqlSession) {
			this.sqlSession.commit(force);
		}
	}
	
	/**
	 * 回滚事务
	 * @param force 当force为true时,事务一定会提交;<br>
	 * 		当force为false时,只有调用过insert、update、delete才会执行提交
	 * @return
	 */
	public void rollbackTransaction(boolean force) {
		if(null != this.sqlSession) {
			this.sqlSession.rollback(force);
		}
	}

	@Override
	public void releaseResource() {
		if(null != sqlSession) {
			sqlSession.close();
			sqlSession = null;
		}
	}
	
	/**
	 * 设置SqlSession对象
	 * @param sqlWriteClient
	 */
	public void setSqlSessionFactory(SqlSessionFactory sqlSessionFactory) {
		this.sqlSessionFactory = sqlSessionFactory;
	}
	
	public SqlSessionFactory getSqlSessionFactory() {
		return this.sqlSessionFactory;
	}
	
	public void setSqlSession(SqlSession sqlSession) {
		this.sqlSession = sqlSession;
	}
	
	public SqlSession getSqlSession() {
		if(null == sqlSession) {
			sqlSession = sqlSessionFactory.openSession(ExecutorType.REUSE, true);
		}
		
		return sqlSession;
	}

	/**
	 * 获取selectType
	 * @return  the selectType
	 * @since   Ver 1.0
	 */
	public int getSelectType() {
		return selectType;
	}

	/**
	 * 设置selectType
	 * @param   selectType    
	 * @since   Ver 1.0
	 */
	public void setSelectType(int selectType) {
		this.selectType = selectType;
	}
}

