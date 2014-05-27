/**
 * DBConnect2.java
 * com.nearme.base.cache
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-19 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache;

import org.apache.ibatis.session.ExecutorType;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;

import com.nearme.base.cache.selector.CacheSelector;

/**
 * ClassName:DBConnect2
 * Function: DB连接
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-10-19  上午11:21:27
 */
public class DBConnect2 extends AbstractCache2 {
	/**
	 * select one模式
	 */
	public static final int SELECT_TYPE_ONE = 0;
	
	/**
	 * select list模式
	 */
	public static final int SELECT_TYPE_LIST = 1;
	
	/**
	 * select map模式
	 */
	public static final int SELECT_TYPE_MAP = 2;

	private SqlSessionFactory sqlSessionFactory;
	private SqlSession sqlSession;
	private ExecutorType executorType;
	private int selectType;
	private boolean autoRelease;
	
	public DBConnect2() {
		this(SELECT_TYPE_ONE, ExecutorType.REUSE);
	}
	
	/**
	 * 
	 * Creates a new instance of DBConnect2.
	 *
	 * @param selectType 查询类型，0查询单个对象，1查询list
	 */
	public DBConnect2(int selectType) {
		this(selectType, ExecutorType.REUSE);
	}
	
	/**
	 * 
	 * Creates a new instance of DBConnect2.
	 *
	 * @param executorType 数据连接类型
	 * @see org.apache.ibatis.session.ExecutorType
	 */
	public DBConnect2(ExecutorType executorType) {
		this(SELECT_TYPE_ONE, executorType);
	}
	
	/**
	 * 
	 * Creates a new instance of DBConnect2.
	 *
	 * @param selectType	查询类型，0查询单个对象，1查询list
	 * @param executorType	数据连接类型
	 */
	public DBConnect2(int selectType, ExecutorType executorType) {
		super();
		this.selectType = selectType;
		this.executorType = executorType;
	}
	
	@Override
	public int deleteCache(CacheSelector cacheSelector) throws Exception {
		try {
			return getSqlSession().delete(cacheSelector.getSelectorKey(), cacheSelector.getSelectorValue());
		} finally {
			releaseOnCondition();
		}
	}

	@Override
	public Object getCache(CacheSelector cacheSelector) throws Exception {
		SqlSession sqlSession = getSqlSession();
		
		try {
			switch(this.selectType) {
			case SELECT_TYPE_ONE:
				return sqlSession.selectOne(cacheSelector.getSelectorKey(), cacheSelector.getSelectorValue());
			case SELECT_TYPE_LIST:
				return sqlSession.selectList(cacheSelector.getSelectorKey(), cacheSelector.getSelectorValue());
			case SELECT_TYPE_MAP:
	//			return sqlSession.selectMap(cacheSelector.getSelectorKey(), cacheSelector.getSelectorValue());
			default:
				return null;
			}
		} finally {
			releaseOnCondition();
		}
	}

	@Override
	public int insertCache(CacheSelector cacheSelector) throws Exception {
		try {
			return getSqlSession().insert(cacheSelector.getSelectorKey(), cacheSelector.getSelectorValue());
		} finally {
			releaseOnCondition();
		}
	}

	@Override
	public int updateCache(CacheSelector cacheSelector) throws Exception {
		try {
			return getSqlSession().update(cacheSelector.getSelectorKey(), cacheSelector.getSelectorValue());
		} finally {
			releaseOnCondition();
		}
	}

	@Override
	public boolean setCache(CacheSelector cacheSelector, Object cacheObject) {
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
		//关闭自动释放资源开关
		this.setAutoRelease(false);
		
		//打开一个
		sqlSession = sqlSessionFactory.openSession(executorType, false);
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
			this.setAutoRelease(true);
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
			this.setAutoRelease(true);
		}
	}
	
	protected void releaseOnCondition() {
		if(this.autoRelease) {
			//如果是自动释放
			this.releaseResource();
		}
	}

	@Override
	public void releaseResource() {
		if(null != sqlSession) {
			sqlSession.close();
			sqlSession = null;
		}
	}

	@Override
	public boolean expireSupported() {
		return false;
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
			sqlSession = sqlSessionFactory.openSession(executorType, true);
		}
		
		return sqlSession;
	}
	
	/**
	 * 获取autoRelease
	 * @return  the autoRelease
	 * @since   Ver 1.0
	 */
	public boolean isAutoRelease() {
		return autoRelease;
	}

	/**
	 * 设置autoRelease
	 * @param   autoRelease    
	 * @since   Ver 1.0
	 */
	public void setAutoRelease(boolean autoRelease) {
		this.autoRelease = autoRelease;
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

