package org.jabe.neverland.model;

import java.util.List;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

public abstract class ListItem<T> implements ListElement<T>,
		View.OnClickListener {

	protected List<T> mLists;
	protected Context mContext;

	public ListItem(List<T> list, Context context) {
		this.mContext = context;
		this.mLists = list;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.jabe.neverland.model.ListElement#getView(int, android.view.View,
	 * android.view.ViewGroup)
	 */
	@Override
	public View getView(int position, View convertView, ViewGroup parent) {
		View view = convertView;
		IViewHolder holder = null;
		if (useConvertView()) {
			// 缓存ViewHolder
			if (view == null) {
				// 耗时操作inflate
				view = LayoutInflater.from(mContext).inflate(getLayoutId(),
						null);
				holder = getViewHolder();
				holder.saveView(view, position, true);
			} else {
				// 获取缓存holder
				holder = (IViewHolder) view.getTag();
				if (holder == null) {
					// 耗时操作inflate
					view = LayoutInflater.from(mContext).inflate(getLayoutId(),
							null);
					holder = getViewHolder();
					holder.saveView(view, position, true);
				} else {
					// 缓存获取成功
				}
			}
		} else {
			// inflate,但是不缓存
			view = LayoutInflater.from(mContext).inflate(getLayoutId(), null);
			holder = getViewHolder();
			holder.saveView(view, position, false);
		}
		// do something for holder~~~
		refreshHolder(holder, position);
		return view;
	}

	public abstract ListElement<T> clone();

	public abstract void click(View v);

	@Override
	public List<T> getDataList() {

		return mLists;
	}

	@Override
	public void remove(int postion) {
		mLists.remove(postion);
	}

	@Override
	public void add(T data) {
		mLists.add(data);
	}

	@Override
	public void insert(T data, int position) {
		mLists.add(position, data);
	}

	@Override
	public void onClick(View v) {
		this.click(v);
	}
}
