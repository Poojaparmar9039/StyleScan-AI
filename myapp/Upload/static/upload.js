function toggleStyleCategory() {
    const styleType = document.getElementById('styleType').value;
    const categoryGroup = document.getElementById('clothingCategoryGroup');
    if (styleType === 'Clothes') {
      categoryGroup.style.display = 'block';
    } else {
      categoryGroup.style.display = 'none';
    }
  }