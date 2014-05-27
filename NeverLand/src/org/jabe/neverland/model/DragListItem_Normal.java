package org.jabe.neverland.model;

import java.util.List;

import org.jabe.neverland.R;
import org.jabe.neverland.model.ViewHolderFactory.DragListViewHolder;

import android.content.Context;
import android.view.View;

public class DragListItem_Normal extends ListItem<String> {

	public DragListItem_Normal(List<String> list, Context context) {
		super(list, context);
	}

	@Override
	public int getLayoutId() {
		return R.layout.drag_list_item_all;
	}

	@Override
	public void click(View v) {
		
	}


	@Override
	public boolean isClickable() {
		return true;
	}


	@Override
	public void refreshHolder(IViewHolder holder, int position) {
		ViewHolderFactory.DragListViewHolder h = (DragListViewHolder) holder;
		h.box1.setVisibility(View.GONE);
		h.box2.setVisibility(View.VISIBLE);
		h.content.setText(mLists.get(position));
	}


	@Override
	public boolean useConvertView() {
		return true;
	}


	@Override
	public IViewHolder getViewHolder() {
		return new ViewHolderFactory.DragListViewHolder();
	}

	@Override
	public ListElement<String> clone() {
		if (canReuse()) {
			return this;	
		} else {
			return new DragListItem_Normal(mLists, mContext);
		}
	}

	@Override
	public boolean canReuse() {
		return true;
	}
}
